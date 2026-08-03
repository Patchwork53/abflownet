import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from contextlib import nullcontext
from tqdm.auto import tqdm

from abflownet.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from abflownet.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from abflownet.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition
from abflownet.modules.common.layers import clampped_one_hot
import random



def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3) 
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        log_Z_mode='conditional',
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

        # TB partition function: 'conditional' (default) or 'global' (paper scalar).
        if log_Z_mode not in ('conditional', 'global'):
            raise ValueError(
                f"log_Z_mode must be 'conditional' or 'global', got {log_Z_mode!r}"
            )
        self.log_Z_mode = log_Z_mode
        if log_Z_mode == 'conditional':
            # log Z_θ(x): per antigen/framework context
            self.log_Z_net = nn.Sequential(
                nn.Linear(res_feat_dim, res_feat_dim),
                nn.SiLU(),
                nn.Linear(res_feat_dim, 1),
            )
            self.log_Z = None
        else:
            # Single scalar Z over the whole distribution (paper)
            self.log_Z = nn.Parameter(torch.zeros(1))
            self.log_Z_net = None

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def _conditional_log_Z(self, res_feat, mask_generate, mask_res):
        """log Z_θ(x) from pooled context residues (not the generated CDR).

        Args:
            res_feat: (N, L, D)
            mask_generate: (N, L) True on designed CDR sites
            mask_res: (N, L) True on valid residues
        Returns:
            log_Z: (N,)
        """
        ctx_mask = mask_res & ~mask_generate
        denom = ctx_mask.sum(dim=1, keepdim=True).clamp(min=1).to(res_feat.dtype)
        pooled = (res_feat * ctx_mask.unsqueeze(-1).to(res_feat.dtype)).sum(dim=1) / denom
        return self.log_Z_net(pooled).squeeze(-1)

    def _compute_log_Z(self, res_feat, mask_generate, mask_res):
        """Return log Z with shape (N,) for the active log_Z_mode."""
        N = res_feat.shape[0]
        if self.log_Z_mode == 'conditional':
            return self._conditional_log_Z(res_feat, mask_generate, mask_res)
        return self.log_Z.expand(N)

    def _trajectory_balance_loss(
        self,
        v_0, p_0, s_0,
        res_feat, pair_feat,
        mask_generate, mask_res,
        denoise_structure, denoise_sequence,
        energies,
    ):
        """Trajectory Balance loss (paper Eq. TB / loss_combo).

        Builds a full noising trajectory under q, then evaluates
        log p_θ(S^{t-1}|S^t) on those same states. Gradients flow through
        one randomly chosen timestep only (paper appendix).
        """
        N = res_feat.shape[0]
        device = v_0.device

        # --- Forward trajectory under q (no model params) ---
        with torch.no_grad():
            v_traj = [v_0.detach()]
            p_traj = [p_0.detach()]
            s_traj = [s_0.detach()]
            forward_logprob = []

            for step in range(1, self.num_steps + 1):
                t_current = torch.full((N,), step, dtype=torch.long, device=device)

                if denoise_structure:
                    v_noisy, _, rot_fp = self.trans_rot.add_noise_step_wise(
                        v_traj[-1], mask_generate, t_current, return_prob=True
                    )
                    p_noisy, _, pos_fp = self.trans_pos.add_noise_step_wise(
                        p_traj[-1], mask_generate, t_current, return_prob=True
                    )
                else:
                    v_noisy, p_noisy = v_traj[-1], p_traj[-1]
                    rot_fp = torch.zeros(N, mask_generate.size(1), device=device)
                    pos_fp = torch.zeros(N, mask_generate.size(1), device=device)

                if denoise_sequence:
                    _, s_noisy, seq_fp = self.trans_seq.add_noise_step_wise(
                        s_traj[-1], mask_generate, t_current, return_prob=True
                    )
                else:
                    s_noisy = s_traj[-1]
                    seq_fp = torch.zeros(N, mask_generate.size(1), device=device)

                v_traj.append(v_noisy)
                p_traj.append(p_noisy)
                s_traj.append(s_noisy)
                # Product over CDR sites → sum of log-probs, shape (N,)
                forward_logprob.append((seq_fp + pos_fp + rot_fp).sum(dim=-1))

            sum_log_q = torch.stack(forward_logprob, dim=1).sum(dim=1)

        # --- Reverse log-probs on the same trajectory ---
        # L_TB = (log Z + Σ log p_θ - log R - Σ log q)^2
        # R(S^0) = exp(-α · BindingEnergy), α = 10^{-2}
        # log Z is either Z_θ(x) (conditional) or a global scalar.
        alpha = 1e-2
        log_R = -alpha * energies
        log_Z = self._compute_log_Z(res_feat, mask_generate, mask_res)

        backward_logprob = []
        step_when_to_backprop = random.randint(1, self.num_steps)

        for step in range(1, self.num_steps + 1):
            v_t, p_t, s_t = v_traj[step], p_traj[step], s_traj[step]
            v_tm1, p_tm1, s_tm1 = v_traj[step - 1], p_traj[step - 1], s_traj[step - 1]

            beta = self.trans_pos.var_sched.betas[step].expand([N])
            t_tensor = torch.full([N], fill_value=step, dtype=torch.long, device=device)

            ctx = nullcontext() if step == step_when_to_backprop else torch.no_grad()
            with ctx:
                v_pred, _, eps_p, c_denoised = self.eps_net(
                    v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
                )
                rot_bp = self.trans_rot.log_prob_denoise(v_tm1, v_pred, mask_generate, t_tensor)
                pos_bp = self.trans_pos.log_prob_denoise(p_t, p_tm1, eps_p, mask_generate, t_tensor)
                seq_bp = self.trans_seq.log_prob_denoise(s_t, s_tm1, c_denoised, mask_generate, t_tensor)
                backward_logprob.append((seq_bp + pos_bp + rot_bp).sum(dim=-1))

        sum_log_p = torch.stack(backward_logprob, dim=1).sum(dim=1)
        residual = log_Z + sum_log_p - log_R - sum_log_q
        return (residual ** 2).mean()

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence, energies, it, start_tb_after):
        N, L = res_feat.shape[:2]
        device = v_0.device
        p_0 = self._normalize_position(p_0)
        loss_dict = {}

        if it > start_tb_after:
            loss_dict['tb'] = self._trajectory_balance_loss(
                v_0, p_0, s_0, res_feat, pair_feat,
                mask_generate, mask_res,
                denoise_structure, denoise_sequence,
                energies,
            )
        else:
            loss_dict['tb'] = torch.zeros((), device=device)

        # DiffAb denoising losses at a random timestep (independent of TB trajectory)
        t_random = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        if denoise_structure:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, eps_rot = self.trans_rot.add_noise(v_0, mask_generate, t_random)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t_random)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t_random)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t_random]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20)


        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t_random)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t_random) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred,
            target=post_true,
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq


        return loss_dict



    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
            
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj