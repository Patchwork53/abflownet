
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abflownet.modules.common.layers import clampped_one_hot
from abflownet.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


def _gaussian_log_prob(diff, std, mask_generate, t=None, drop_t_eq_1=False):
    """Isotropic Gaussian log-density on the last dimension (size 3).

    Continuous densities may be > 1, so log-prob can be positive — do not clamp to 0.
    When drop_t_eq_1 is set, t=1 (sigma=0, deterministic reverse step) contributes 0.
    """
    # Replace non-finite diffs (e.g. SO(3) log-map edge cases) before /var.
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    std_safe = std.clamp_min(1e-5)
    var = std_safe ** 2
    log_prob = (
        -0.5 * torch.sum(diff ** 2 / var, dim=-1)
        - 0.5 * 3 * torch.log(torch.tensor(2 * np.pi, device=diff.device, dtype=diff.dtype))
        - 3 * torch.log(std_safe.squeeze(-1))
    )
    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=50.0, neginf=-50.0)
    if drop_t_eq_1:
        log_prob = torch.where((t > 1)[:, None], log_prob, torch.zeros_like(log_prob))
    return torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))


def _relative_so3vec(v_sample, v_center):
    """so3vec of the relative rotation E in R_sample = E @ R_center (DiffAb composition)."""
    R_sample = so3vec_to_rotation(v_sample)
    R_center = so3vec_to_rotation(v_center)
    R_rel = R_sample @ R_center.transpose(-1, -2)
    return rotation_to_so3vec(R_rel)


class PositionTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(p_t|p_0)
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        if return_prob:
            diff = p_noisy - c0*p_0
            log_prob = _gaussian_log_prob(diff, c1, mask_generate)
            return p_noisy, e_rand, log_prob
        else:
            return p_noisy, e_rand

    def add_noise_step_wise(self, p_t_1, mask_generate, t, return_prob=False):
        """
        Forward transition q(p_t|p_{t-1}) = N(p_t | sqrt(1-β)·p_{t-1}, β·I)
        """
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(1-beta).view(-1, 1, 1)
        c1 = torch.sqrt(beta).view(-1, 1, 1)

        e_rand = torch.randn_like(p_t_1)
        p_noisy = c0*p_t_1 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_t_1), p_noisy, p_t_1)

        if return_prob:
            diff = p_noisy - c0*p_t_1
            log_prob = _gaussian_log_prob(diff, c1, mask_generate)
            return p_noisy, e_rand, log_prob
        else:
            return p_noisy, e_rand


    def _denoise_params(self, p_t, eps_p, t):
        # Match DiffAb sampling: reverse variance is the DDPM posterior sigma, not beta.
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, 1, 1)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, 1, 1)
        mu = c0 * (p_t - c1 * eps_p)
        return mu, sigma

    def log_prob_denoise(self, p_t, p_tm1, eps_p, mask_generate, t):
        """log p_θ(p_{t-1}|p_t) evaluated at the trajectory state p_tm1."""
        mu, sigma = self._denoise_params(p_t, eps_p, t)
        return _gaussian_log_prob(p_tm1 - mu, sigma, mask_generate, t=t, drop_t_eq_1=True)

    def denoise(self, p_t, eps_p, mask_generate, t, return_prob=False):
        mu, sigma = self._denoise_params(p_t, eps_p, t)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = mu + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)

        if return_prob:
            log_prob = self.log_prob_denoise(p_t, p_next, eps_p, mask_generate, t)
            return p_next, log_prob
        else:
            return p_next


class RotationTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        beta = self.var_sched.betas
        self.angular_distrib_step_wise = ApproxAngularDistribution(beta.tolist(), **angular_distrib_fwd_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(v_t|v_0)
        """
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)
        R0_scaled = so3vec_to_rotation(c0 * v_0)
        E_scaled = so3vec_to_rotation(e_scaled)
        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        if return_prob:
            # Density of the composed noise E, not a Euclidean difference of so3vecs.
            e_rel = _relative_so3vec(v_noisy, c0 * v_0)
            log_prob = _gaussian_log_prob(e_rel, c1, mask_generate)
            return v_noisy, e_scaled, log_prob
        else:
            return v_noisy, e_scaled


    def add_noise_step_wise(self, v_t_1, mask_generate, t, return_prob=False):
        """
        Step-wise forward transition q(v_t|v_{t-1})
        """
        N, L = mask_generate.size()
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(1-beta).view(-1, 1, 1)
        c1 = torch.sqrt(beta).view(-1, 1, 1)

        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_step_wise, device=self._dummy.device)
        Rt_scaled = so3vec_to_rotation(c0 * v_t_1)
        E_scaled = so3vec_to_rotation(e_scaled)
        R_noisy = E_scaled @ Rt_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_t_1), v_noisy, v_t_1)

        if return_prob:
            e_rel = _relative_so3vec(v_noisy, c0 * v_t_1)
            log_prob = _gaussian_log_prob(e_rel, c1, mask_generate)
            return v_noisy, e_scaled, log_prob
        else:
            return v_noisy, e_scaled


    def log_prob_denoise(self, v_tm1, v_pred, mask_generate, t):
        """log p_θ(v_{t-1}|v_t) at trajectory state v_tm1.

        DiffAb samples R_{t-1} = E @ R_pred with E ~ IG(sigma). Evaluate the
        Euclidean-on-so3vec approximation on that relative rotation E.
        """
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)
        e_rel = _relative_so3vec(v_tm1, v_pred)
        return _gaussian_log_prob(e_rel, sigma, mask_generate, t=t, drop_t_eq_1=True)

    def denoise(self, v_t, v_next, mask_generate, t, return_prob=False):
        """
        Denoising step with optional probability calculation
        """
        N, L = mask_generate.size()

        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e)
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next_noisy = rotation_to_so3vec(R_next)
        v_next_noisy = torch.where(mask_generate[..., None].expand_as(v_next_noisy), v_next_noisy, v_t)

        if return_prob:
            log_prob = self.log_prob_denoise(v_next_noisy, v_next, mask_generate, t)
            return v_next_noisy, log_prob
        else:
            return v_next_noisy


class AminoacidCategoricalTransition(nn.Module):
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        N, L, K = c.size()
        c_ = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c_, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(x_t|x_0) for categorical data
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]
        c_noisy = (alpha_bar*c_0) + ((1-alpha_bar)/K)
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_0)
        x_t = self._sample(c_t)

        if return_prob:
            idx = x_t.unsqueeze(-1)
            p_x = torch.gather(c_t, dim=-1, index=idx).squeeze(-1)
            log_p_x = torch.log(p_x+1e-8)
            log_p_x = torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))
            return c_t, x_t, log_p_x
        else:
            return c_t, x_t

    def add_noise_step_wise(self, x_t_1, mask_generate, t, return_prob=False):
        """
        Step-wise forward transition q(x_t|x_{t-1}) for categorical data

        q(x_j^t|x_j^{t-1}) = Multinomial((1 - β_t) · onehot(x_j^{t-1}) + β_t/K)
        """
        N, L = x_t_1.size()
        K = self.num_classes

        c_t_1 = clampped_one_hot(x_t_1, num_classes=K).float()
        beta_t = self.var_sched.betas[t][:, None, None]
        c_noisy = (1 - beta_t) * c_t_1 + (beta_t/K)
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_t_1)
        x_t = self._sample(c_t)

        if return_prob:
            idx = x_t.unsqueeze(-1)
            p_x = torch.gather(c_t, dim=-1, index=idx).squeeze(-1)
            log_p_x = torch.log(p_x + 1e-8)
            log_p_x = torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))
            return c_t, x_t, log_p_x
        else:
            return c_t, x_t


    def posterior(self, x_t, x_0, t):
        """
        Posterior distribution q(x_{t-1}|x_t, x_0) used by DiffAb sampling.

        Note: DiffAb uses alpha_bars[t] for both factors (legacy). Kept for
        consistency between TB log-probs and the actual reverse sampler.
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()

        if x_0.dim() == 3:
            c_0 = x_0
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()

        alpha = self.var_sched.alpha_bars[t][:, None, None]
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def log_prob_denoise(self, x_t, x_tm1, c_0_pred, mask_generate, t):
        """log p_θ(x_{t-1}|x_t) evaluated at the trajectory state x_tm1."""
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()
        post = self.posterior(c_t, c_0_pred, t=t)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        # Non-CDR / padding residues may have aa ids outside [0, 19]; only CDR
        # positions contribute to the log-prob.
        idx = torch.where(mask_generate, x_tm1, torch.zeros_like(x_tm1)).unsqueeze(-1)
        p_x = torch.gather(post, dim=-1, index=idx).squeeze(-1)
        log_p_x = torch.log(p_x + 1e-8)
        return torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))

    def denoise(self, x_t, c_0_pred, mask_generate, t, return_prob=False):
        """
        Denoising step with option to return probability information
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()
        post = self.posterior(c_t, c_0_pred, t=t)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)

        if return_prob:
            log_p_x = self.log_prob_denoise(x_t, x_next, c_0_pred, mask_generate, t)
            return post, x_next, log_p_x
        else:
            return post, x_next
