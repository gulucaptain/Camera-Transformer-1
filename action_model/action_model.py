"""
action_model.py

"""
from action_model.models import DiT
from action_model import create_diffusion
from . import gaussian_diffusion as gd
import torch
from torch import nn
import torch.nn.functional as F

def haar_dwt_1d(x, levels=3):
    """
    x: [B, T, C], T should be divisible by 2**levels (or we pad).
    returns:
      a: low-frequency at last level [B, T/2**L, C]
      ds: list of detail coeffs [d_L, d_{L-1}, ..., d_1]
    """
    B, T, C = x.shape
    # pad T to be divisible by 2**levels
    m = 2 ** levels
    pad = (m - (T % m)) % m
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad), mode="replicate")  # pad on time dim
    a = x
    ds = []
    for _ in range(levels):
        # split even/odd
        even = a[:, 0::2, :]
        odd  = a[:, 1::2, :]
        # Haar low/high
        low  = (even + odd) * 0.5
        high = (even - odd) * 0.5
        ds.append(high)
        a = low
    # ds currently [d1, d2, ...]; reverse to [dL ... d1] if you like
    return a, ds[::-1], pad

def wavelet_multiscale_l1(x_hat, x, levels=3, lambdas=None):
    """
    x_hat, x: [B,T,C]
    lambdas: list length levels+1 -> [lambda_a, lambda_dL, ..., lambda_d1]
    """
    a_h, ds_h, pad_h = haar_dwt_1d(x_hat, levels=levels)
    a,   ds,   pad   = haar_dwt_1d(x,     levels=levels)
    assert pad_h == pad

    if lambdas is None:
        # example: emphasize higher-frequency details a bit more
        # lambdas = [lambda_a, lambda_dL, ..., lambda_d1]
        lambdas = [1.0] + [1.0] * levels

    loss = lambdas[0] * (a_h - a).abs().mean()
    for i in range(levels):
        loss = loss + lambdas[i+1] * (ds_h[i] - ds[i]).abs().mean()
    return loss


# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class ActionModel(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 100,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        # new loss
        x0_hat = self.diffusion._predict_xstart_from_eps(x_t, timestep, noise_pred)
        loss_wav = wavelet_multiscale_l1(x0_hat, x, levels=3, lambdas=[1.0, 1.0, 1.0, 1.0])
        loss_wav = loss_wav * 0.05

        loss = loss + loss_wav
        
        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = self.noise_schedule,
                                               diffusion_steps = self.diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion
