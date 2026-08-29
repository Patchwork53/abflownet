import os
import shutil
import argparse
import torch
import pickle
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from abflownet.datasets import get_dataset
from abflownet.models import get_model
from abflownet.utils.misc import *
from abflownet.utils.data import *
from abflownet.utils.train import *


def _validate_binding_energies(energies_dict, logger):
    """Paper reward assumes InterfaceAnalyzer ΔG roughly in [-100, 0]."""
    vals = list(energies_dict.values())
    if not vals:
        raise ValueError('precomputed_energies.pkl is empty.')
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted) // 2]
    frac_in_range = sum(1 for v in vals if -100.0 <= v <= 0.0) / len(vals)
    logger.info(
        'Binding energies: n=%d min=%.2f median=%.2f max=%.2f frac_in_[-100,0]=%.3f',
        len(vals), vals_sorted[0], median, vals_sorted[-1], frac_in_range,
    )
    if median > 50.0 or frac_in_range < 0.2:
        logger.warning(
            'precomputed_energies.pkl does not look like InterfaceAnalyzer ΔG '
            '(expected roughly [-100, 0]). The bundled pickle appears to store '
            'total Rosetta scores instead. Re-run:\n'
            '  python precompute_binding_energies.py --chothia_dir data/all_structures/chothia '
            '--summary_path data/sabdab_summary_all.tsv --out precomputed_energies.pkl'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--energies', type=str, default='precomputed_energies.pkl')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        log_dir = None
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Energies (filter dataset before building DataLoaders)
    pkl_path = args.energies
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f'Pickle file {pkl_path} not found. Generate it with '
            'precompute_binding_energies.py (InterfaceAnalyzer ΔG).'
        )
    with open(pkl_path, 'rb') as f:
        precomputed_energies = pickle.load(f)
    _validate_binding_energies(precomputed_energies, logger)

    def _has_energy(pdb_id, energy_table):
        return pdb_id in energy_table or pdb_id.split('_')[0] in energy_table

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    # Drop complexes without ΔG up front. Crashing at start_tb_after is useless;
    # silently using 0.0 would corrupt the TB reward (R=exp(0)=1).
    n_train_before = len(train_dataset)
    train_dataset.ids_in_split = [
        i for i in train_dataset.ids_in_split if _has_energy(i, precomputed_energies)
    ]
    n_val_before = len(val_dataset)
    val_dataset.ids_in_split = [
        i for i in val_dataset.ids_in_split if _has_energy(i, precomputed_energies)
    ]
    logger.info(
        'Energy filter: train %d -> %d | val %d -> %d',
        n_train_before, len(train_dataset), n_val_before, len(val_dataset),
    )
    if len(train_dataset) == 0:
        raise RuntimeError('No training complexes left after energy filter.')

    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(
        val_dataset, batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers,
    )
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    z_optimizer_cfg = config.train.get('z_optimizer', None)
    z_named_params = [
        (name, param) for name, param in model.named_parameters()
        if name.startswith('diffusion.log_Z')
    ]
    z_param_ids = {id(param) for _, param in z_named_params}
    policy_params = [
        param for param in model.parameters() if id(param) not in z_param_ids
    ]
    z_params = [param for _, param in z_named_params]
    if z_optimizer_cfg is not None:
        optimizer = get_optimizer(config.train.optimizer, policy_params)
        z_optimizer = get_optimizer(z_optimizer_cfg, z_params)
        logger.info(
            'Separate Z optimizer: params=%s lr=%.6g',
            [name for name, _ in z_named_params],
            z_optimizer.param_groups[0]['lr'],
        )
    else:
        optimizer = get_optimizer(config.train.optimizer, model)
        z_optimizer = None
        policy_params = list(model.parameters())
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    if z_optimizer is not None:
        z_optimizer.zero_grad()
    it_first = 1
    calibration_cfg = config.train.get('z_calibration', None)
    calibration_enabled = bool(
        calibration_cfg is not None and calibration_cfg.get('enabled', False)
    )
    z_calibrated = not calibration_enabled

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration'] + 1
        # strict=False: allow new heads (e.g. log_Z_bias) after architecture tweaks
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing or unexpected:
            logger.warning('load_state_dict strict=False; missing=%s unexpected=%s', missing, unexpected)
        if args.resume is not None:
            try:
                logger.info('Resuming optimizer states...')
                optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError as e:
                if z_optimizer is None:
                    raise
                # Legacy checkpoints used one optimizer containing policy and Z
                # parameters. Load it once, then retain only policy Adam state.
                logger.info('Loading legacy joint optimizer state: %s', e)
                legacy_optimizer = get_optimizer(config.train.optimizer, model)
                legacy_optimizer.load_state_dict(ckpt['optimizer'])
                for key, value in legacy_optimizer.param_groups[0].items():
                    if key != 'params':
                        optimizer.param_groups[0][key] = value
                for param in policy_params:
                    if param in legacy_optimizer.state:
                        optimizer.state[param] = legacy_optimizer.state[param]
                logger.info(
                    'Transferred Adam state for %d policy parameters.',
                    len(optimizer.state),
                )
            if z_optimizer is not None and ckpt.get('z_optimizer') is not None:
                logger.info('Resuming Z optimizer states...')
                z_optimizer.load_state_dict(ckpt['z_optimizer'])
            logger.info('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
        z_calibrated = bool(ckpt.get('z_calibrated', not calibration_enabled))

    def get_energies(batch_pdb_ids, energy_table):
        """Look up InterfaceAnalyzer ΔG for each complex (entry id or pdb code)."""
        energies = []
        for pdb_id in batch_pdb_ids:
            if pdb_id in energy_table:
                energies.append(energy_table[pdb_id])
            else:
                energies.append(energy_table[pdb_id.split('_')[0]])
        return torch.tensor(energies, dtype=torch.float32, device=args.device)

    # Train
    def train(it, energy_table):
        time_start = current_milli_time()
        model.train()

        batch = recursive_to(next(train_iterator), args.device)
        loss_dict = model(
            batch,
            energies=get_energies(batch['pdb_id'], energy_table),
            it=it,
            start_tb_after=config.train.start_tb_after,
        )
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        loss.backward()
        orig_grad_norm = clip_grad_norm_(policy_params, config.train.max_grad_norm)
        if z_optimizer is not None:
            z_grad_norm = clip_grad_norm_(
                z_params,
                config.train.get('z_max_grad_norm', config.train.max_grad_norm),
            )
        else:
            z_grad_norm = torch.zeros((), device=args.device)
        optimizer.step()
        if z_optimizer is not None:
            z_optimizer.step()
        optimizer.zero_grad()
        if z_optimizer is not None:
            z_optimizer.zero_grad()
        time_backward_end = current_milli_time()

        diagnostics = dict(model.diffusion.last_tb_diagnostics)
        diagnostics.update({
            'grad': orig_grad_norm,
            'z_grad': z_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'z_lr': z_optimizer.param_groups[0]['lr'] if z_optimizer is not None else 0.0,
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        log_losses(loss_dict, it, 'train', logger, writer, others=diagnostics)

        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'z_optimizer': z_optimizer.state_dict() if z_optimizer is not None else None,
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'z_calibrated': z_calibrated,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    def calibrate_log_Z(energy_table):
        """Center the mean unclamped TB residual by shifting log_Z_bias."""
        num_batches = int(calibration_cfg.get('num_batches', 8))
        if num_batches <= 0:
            raise ValueError('train.z_calibration.num_batches must be positive.')

        model.eval()
        residual_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for _ in range(num_batches):
                batch = recursive_to(next(train_iterator), args.device)
                model(
                    batch,
                    energies=get_energies(batch['pdb_id'], energy_table),
                    it=config.train.start_tb_after + 1,
                    start_tb_after=config.train.start_tb_after,
                )
                diagnostics = model.diffusion.last_tb_diagnostics
                batch_size = int(batch['aa'].size(0))
                residual_sum += float(diagnostics['tb_residual_mean']) * batch_size
                sample_count += batch_size

        residual_mean = residual_sum / sample_count
        bias = model.diffusion.log_Z_bias
        bias_before = float(bias.detach())
        with torch.no_grad():
            bias.sub_(residual_mean)
        logger.info(
            'Z calibration: batches=%d samples=%d residual_mean=%.6f '
            'bias_before=%.6f bias_after=%.6f',
            num_batches, sample_count, residual_mean,
            bias_before, float(bias.detach()),
        )
        model.train()
        optimizer.zero_grad()
        if z_optimizer is not None:
            z_optimizer.zero_grad()

    # Validate (diffusion losses only — skip expensive TB rollouts)
    def validate(it, energy_table):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict = model(
                    batch,
                    energies=get_energies(batch['pdb_id'], energy_table),
                    it=it,
                    start_tb_after=float('inf'),
                )
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            if not z_calibrated and it > config.train.start_tb_after:
                calibrate_log_Z(precomputed_energies)
                z_calibrated = True
            train(it, energy_table=precomputed_energies)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it, energy_table=precomputed_energies)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'z_optimizer': z_optimizer.state_dict() if z_optimizer is not None else None,
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                        'z_calibrated': z_calibrated,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
