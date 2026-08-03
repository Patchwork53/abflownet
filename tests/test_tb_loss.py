"""Smoke tests for the Trajectory Balance objective."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abflownet.modules.diffusion.dpm_full import FullDPM


def _make_batch(num_steps=8, N=2, L=6, cdr_len=3, log_Z_mode='conditional'):
    torch.manual_seed(0)
    model = FullDPM(
        res_feat_dim=16,
        pair_feat_dim=8,
        num_steps=num_steps,
        eps_net_opt={'num_layers': 2},
        log_Z_mode=log_Z_mode,
    )
    v_0 = torch.randn(N, L, 3)
    p_0 = torch.randn(N, L, 3) * 5
    s_0 = torch.randint(0, 20, (N, L))
    res_feat = torch.randn(N, L, 16)
    pair_feat = torch.randn(N, L, L, 8)
    mask_generate = torch.zeros(N, L, dtype=torch.bool)
    mask_generate[:, :cdr_len] = True
    mask_res = torch.ones(N, L, dtype=torch.bool)
    energies = torch.tensor([-20.0, -5.0])
    return model, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, energies


def test_tb_conditional_z_grad():
    model, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, energies = _make_batch(
        log_Z_mode='conditional',
    )
    assert model.log_Z_mode == 'conditional'
    assert model.log_Z_net is not None
    assert model.log_Z is None

    loss_dict = model(
        v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
        denoise_structure=True, denoise_sequence=True,
        energies=energies, it=1, start_tb_after=0,
    )
    tb = loss_dict['tb']
    assert tb.ndim == 0 and torch.isfinite(tb) and tb.item() >= 0.0

    tb.backward()
    z_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.log_Z_net.parameters()
    )
    assert z_grad > 0
    eps_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.eps_net.parameters()
    )
    assert eps_grad > 0

    with torch.no_grad():
        z1 = model._compute_log_Z(res_feat, mask_generate, mask_res)
        z2 = model._compute_log_Z(res_feat + 1.0, mask_generate, mask_res)
        assert z1.shape == (res_feat.size(0),)
        assert not torch.allclose(z1, z2)


def test_tb_global_z_grad():
    model, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, energies = _make_batch(
        log_Z_mode='global',
    )
    assert model.log_Z_mode == 'global'
    assert model.log_Z is not None
    assert model.log_Z_net is None

    loss_dict = model(
        v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
        denoise_structure=True, denoise_sequence=True,
        energies=energies, it=1, start_tb_after=0,
    )
    tb = loss_dict['tb']
    assert tb.ndim == 0 and torch.isfinite(tb)

    tb.backward()
    assert model.log_Z.grad is not None
    assert model.log_Z.grad.abs().sum().item() > 0

    with torch.no_grad():
        z = model._compute_log_Z(res_feat, mask_generate, mask_res)
        assert z.shape == (res_feat.size(0),)
        assert torch.allclose(z, model.log_Z.expand_as(z))


def test_tb_default_is_conditional():
    model = FullDPM(res_feat_dim=8, pair_feat_dim=4, num_steps=4, eps_net_opt={'num_layers': 1})
    assert model.log_Z_mode == 'conditional'


def test_tb_disabled_before_schedule():
    model, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, energies = _make_batch()
    loss_dict = model(
        v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
        denoise_structure=True, denoise_sequence=True,
        energies=energies, it=10, start_tb_after=100,
    )
    assert loss_dict['tb'].item() == 0.0


def test_reward_term_matches_paper():
    alpha = 1e-2
    E = torch.tensor([-20.0, 0.0, 50.0])
    log_R = -alpha * E
    assert torch.allclose(log_R, torch.tensor([0.2, 0.0, -0.5]))
    assert math.isclose(torch.exp(log_R[0]).item(), math.exp(0.2), rel_tol=1e-6)


if __name__ == '__main__':
    test_reward_term_matches_paper()
    test_tb_disabled_before_schedule()
    test_tb_default_is_conditional()
    test_tb_conditional_z_grad()
    test_tb_global_z_grad()
    print('All TB smoke tests passed.')
