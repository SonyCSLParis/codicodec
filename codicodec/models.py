from .hparams import *
from .utils import get_sigma_continuous, get_c, reverse_step, distribute, preprocess_parallel_input, postprocess_parallel_input, preprocess_parallel_features

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, Transformer_Diffusion


if torch_compile_cache_dir is not None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = torch_compile_cache_dir

torch.backends.cudnn.benchmark = True


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3):
        super().__init__()
        self._levels = torch.tensor(levels)
        self._eps = eps
        self._basis = torch.cat([torch.tensor([1]), torch.cumprod(self._levels[:-1], dim=0)])
        self._basis = self._basis.to(torch.int64)

        self.register_buffer('_levels_tensor', self._levels)
        self.register_buffer('_basis_tensor', self._basis)

        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return self._levels.prod().item()

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_tensor - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels_tensor % 2 == 1, 0., 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_tensor // 2
        return quantized / half_width
    
    def dont_quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Does not quantize z, returns unquantized zhat, same shape as z."""
        not_quantized = self.bound(z)

        # Renormalize to [-1, 1].
        half_width = self._levels_tensor // 2
        return not_quantized / half_width
    
    def round_continuous(self, z: torch.Tensor) -> torch.Tensor:
        half_width = self._levels_tensor.to(z.device) // 2
        z = z * half_width
        zhat = torch.round(z)
        return zhat / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_tensor // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_tensor // 2
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis_tensor).sum(dim=-1).to(torch.int64)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `indexes_to_codes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.div(indices, self._basis_tensor, rounding_mode='floor') % self._levels_tensor
        return self._scale_and_shift_inverse(codes_non_centered)


def exists(val):
    return val is not None

def init(module):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.)
    return module

def zero_init(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1,1,freq_dim,1)))

    def forward(self, input):
        return input*self.scale


class GroupNorm(nn.Module):
    def __init__(self, dim, cond_dim=None, affine=True):
        super(GroupNorm, self).__init__()
        self.affine = affine
        self.norm = nn.GroupNorm(min(dim//4, 32), dim, affine=False)
        if exists(cond_dim):
            self.cond_proj = zero_init(nn.Linear(cond_dim, dim))
        else:
            if affine:
                self.weight = nn.Parameter(torch.ones((dim,)))

    def forward(self, x, cond=None):
        x = self.norm(x)
        if exists(cond):
            cond = self.cond_proj(cond)
            if x.dim()==4:
                cond = cond.view(cond.shape[0], -1, 1, 1)
            else:
                cond = cond.view(cond.shape[0], -1, 1)
            x = x * (1.+cond)
        else:
            if self.affine:
                if x.dim()==4:
                    x = x * self.weight.view(1,-1,1,1)
                else:
                    x = x * self.weight.view(1,-1,1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, cond_dim=None, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if exists(cond_dim):
            self.cond_proj = zero_init(nn.Linear(cond_dim, dim))
        else:
            if affine:
                self.weight = nn.Parameter(torch.ones((dim,)))

    def forward(self, x, cond=None):
        x = (x.float() * torch.rsqrt(x.float().pow(2).mean(-2, keepdim=True) + self.eps)).type_as(x)
        if exists(cond):
            cond = self.cond_proj(cond)
            cond = cond.view(cond.shape[0], -1, 1)
            x = x * (1.+cond)
        else:
            if self.affine:
                x = x * self.weight.view(1,-1,1)
        return x
    

class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor):
        super(Downsample, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.groups = input_channels

        # Initialize the weights to average filter
        weight = torch.ones(output_channels, input_channels // self.groups, *factor)
        weight = weight / (factor[0] * factor[1])
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=self.factor, padding=0, groups=self.groups)
        return x
    

class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor):
        super(Upsample, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.groups = output_channels

        # Initialize the weights for evenly spreading the values
        weight = torch.ones(input_channels, output_channels // self.groups, *factor)
        weight = weight / (input_channels/output_channels)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = F.conv_transpose2d(x, self.weight, stride=self.factor, padding=0, groups=self.groups)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(UpsampleConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(2,2))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp
    

class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DownsampleConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(2,2))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(UpsampleFreqConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(factor,1))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(factor,1), stride=(factor,1), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(DownsampleFreqConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(factor,1))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(factor,1), stride=(factor,1), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class UpsampleTimeConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(UpsampleTimeConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up = Upsample(in_channels, out_channels, factor=(1,factor))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,factor), stride=(1,factor), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.up(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class DownsampleTimeConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, factor=4):
        super(DownsampleTimeConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down = Downsample(in_channels, out_channels, factor=(1,factor))
        self.norm = GroupNorm(in_channels)
        self.c = zero_init(nn.Conv2d(in_channels, out_channels, kernel_size=(1,factor), stride=(1,factor), padding=(0,0), bias=False))

    def forward(self, x):
        inp = x.clone()
        inp = self.down(inp)
        x = self.norm(x)
        x = self.c(x)
        return x+inp


class Feedforward(nn.Module):
    def __init__(self, dim, mlp_mult = 1, use_2d=True):
        super().__init__()
        inner_dim = int(dim * mlp_mult)
        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d

        self.ff1 = init(Conv(dim, inner_dim, 3, padding=1, bias=False))
        self.activation = nn.SiLU()
        self.ff2 = zero_init(Conv(inner_dim, dim, 3, padding=1, bias=False))

    def forward(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_mult=1, cond_dim=None, use_2d=True):
        super(ConvBlock, self).__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, use_2d=use_2d)
        self.norm = GroupNorm(dim, cond_dim)

    def forward(self, x, cond=None):
        inp = x.clone()
        x = self.norm(x, cond)
        x = self.ff(x)
        return x+inp


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        freqs = freqs.to(x.dtype)
        # take outer product
        x = x.unsqueeze(-1) * freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class DownFrontend(nn.Module):
    def __init__(self, frontend_layers_list, cond_dim=None):
        super(DownFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        
        input_channels = frontend_base_channels*frontend_multipliers_list[0]

        self.conv_inp = init(nn.Conv2d(stft_channels, input_channels, kernel_size=3, stride=1, padding=1))
        
        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(self.frontend_layers_list,frontend_multipliers_list)):
            output_channels = frontend_base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ConvBlock(output_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                if frontend_freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(output_channels, frontend_base_channels*frontend_multipliers_list[i+1], factor=4))
                elif frontend_freq_downsample_list[i]==2:
                    down_layers.append(DownsampleFreqConv(output_channels, frontend_base_channels*frontend_multipliers_list[i+1], factor=2))
                elif frontend_freq_downsample_list[i]==3:
                    down_layers.append(DownsampleTimeConv(output_channels, frontend_base_channels*frontend_multipliers_list[i+1], factor=2))
                else:
                    down_layers.append(DownsampleConv(output_channels, frontend_base_channels*frontend_multipliers_list[i+1]))
            
        self.down_layers = nn.ModuleList(down_layers)

    def add_feature(self, x, features, index):
        if features is not None:
            x = (x + features[index])/math.sqrt(2.)
        return x

    def forward(self, x, cond=None, features=None, gain=None, log_magnitude=False):

        x = x.to(memory_format=torch.channels_last)
        if features is not None:
            features = [el.to(memory_format=torch.channels_last) for el in features]
        if gain is not None:
            gain = gain.to(memory_format=torch.channels_last)

        x = self.conv_inp(x)

        if gain is not None:
            x = x*gain
        
        # DOWNSAMPLING
        new_features = []
        k = 0
        k_feat = 0
        for i,num_layers in enumerate(self.frontend_layers_list):
            for num in range(num_layers):
                x = self.add_feature(x, features, k_feat)
                k_feat = k_feat+1
                x = self.down_layers[k](x, cond)
                if log_magnitude:
                    print(f'Enc 2D Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1
                new_features.append(x)
            if i!=(len(self.frontend_layers_list)-1):
                x = self.down_layers[k](x)
                if log_magnitude:
                    print(f'Enc 2D Down Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1
                
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # shape [batch, freq*time, dim]

        x = x.to(memory_format=torch.contiguous_format)
        new_features = [el.to(memory_format=torch.contiguous_format) for el in new_features]

        return x, new_features[::-1]
    

class UpFrontend(nn.Module):
    def __init__(self, frontend_layers_list, cond_dim=None):
        super(UpFrontend, self).__init__()

        self.frontend_layers_list = frontend_layers_list
        
        input_channels = frontend_base_channels*frontend_multipliers_list[-1]

        self.freq_dim = (hop*(fac//2))//(4**frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(2))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(frontend_multipliers_list))[1:]+list(reversed(frontend_multipliers_list))[:1]
        freq_upsample_list = list(reversed(frontend_freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(self.frontend_layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ConvBlock(input_channels, cond_dim=cond_dim))
            if i!=(len(self.frontend_layers_list)-1):
                output_channels = frontend_base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels, factor=4))
                elif freq_upsample_list[i]==2:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels, factor=2))
                elif freq_upsample_list[i]==3:
                    up_layers.append(UpsampleTimeConv(input_channels, output_channels, factor=2))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels))
                input_channels = output_channels
            
        self.up_layers = nn.ModuleList(up_layers)

        self.norm_out = GroupNorm(input_channels, cond_dim=cond_dim)
        self.conv_out = zero_init(nn.Conv2d(input_channels, stft_channels, kernel_size=3, stride=1, padding=1))

    def add_feature(self, x, features, index):
        if features is not None:
            x = (x + features[index])/math.sqrt(2.)
        return x

    def forward(self, x, cond=None, features=None, gain=None, skip_output_layer=False, log_magnitude=False):

        x = x.permute(0,2,1).reshape(x.shape[0], dim, self.freq_dim, -1) # shape [batch, dim, freq, time]

        x = x.to(memory_format=torch.channels_last)
        if features is not None:
            features = [el.to(memory_format=torch.channels_last) for el in features]
        if gain is not None:
            gain = gain.to(memory_format=torch.channels_last)

        # UPSAMPLING
        new_features = []
        k = 0
        k_feat = 0
        for i,num_layers in enumerate(reversed(self.frontend_layers_list)):
            for num in range(num_layers):
                x = self.add_feature(x, features, k_feat)
                k_feat = k_feat+1
                x = self.up_layers[k](x, cond)
                if log_magnitude:
                    print(f'Dec 2D Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1
                new_features.append(x)
            if i!=(len(self.frontend_layers_list)-1):
                x = self.up_layers[k](x)
                if log_magnitude:
                    print(f'Dec 2D Up Level {i} Layer {k}: {x.abs().mean()}')
                k = k+1

        if not skip_output_layer:
            x = self.norm_out(x, cond)
            if gain is not None:
                x = x*gain
            x = self.conv_out(x) # shape [batch, stft_channels, freq, time]

        x = x.to(memory_format=torch.contiguous_format)
        new_features = [el.to(memory_format=torch.contiguous_format) for el in new_features]

        return x, new_features[::-1]


class UNet(nn.Module):
    """Consistency Transformer autoencoder used by CodiCodec.

    Provides an encoder that maps spectrogram patches to latents and a decoder
    that reconstructs spectrograms conditioned on noise level. Supports both
    parallel and autoregressive decoding schedules and a finite scalar quantizer
    (FSQ) bottleneck.
    """
    def __init__(self):
        super(UNet, self).__init__()

        self.freq_dim = (hop*(fac//2))//(4**frontend_freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(0))
        self.freq_dim = self.freq_dim//(2**frontend_freq_downsample_list.count(2))

        self.time_dim = spec_length//(2**frontend_freq_downsample_list.count(0))
        self.time_dim = self.time_dim//(2**frontend_freq_downsample_list.count(3))

        scale = float(dim) ** -0.5
        self.emb = PositionalEmbedding(embedding_size=cond_channels)
        self.emb_proj = nn.Sequential(init(nn.Linear(cond_channels, cond_channels)), nn.SiLU(), init(nn.Linear(cond_channels, cond_channels)), nn.SiLU(), init(nn.Linear(cond_channels, cond_channels)), nn.SiLU())
        
        self.latents = nn.Parameter(scale*torch.randn(1, num_latents, dim), requires_grad=True)
        self.mask_embedding = nn.Parameter(scale*torch.randn(1, self.freq_dim, 1, dim), requires_grad=True)

        self.gain_encoder = nn.Parameter(torch.ones(1, 1, hop*(fac//2), 1), requires_grad=True)
        self.gain_decoder = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(), nn.Linear(cond_channels, cond_channels), nn.SiLU(), zero_init(nn.Linear(cond_channels, hop*2*(fac//2))))

        self.frontend_encoder_down = DownFrontend(frontend_encoder_layers_list).to(memory_format=torch.channels_last)
        if num_more_latents>0:
            self.more_latents_encoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_encoder = None
        self.encoder = Transformer(dim, bottleneck_channels, training_length=data_length+num_latents+num_more_latents, dim=dim, num_layers=num_layers_encoder, heads=heads, mlp_mult=mlp_mult, pos_emb=pos_emb)

        self.lat2patch_pre_decoder = init(nn.Linear(bottleneck_channels, dim))
        if num_more_latents>0:
            self.more_latents_pre_decoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_pre_decoder = None
        self.pre_decoder = Transformer(dim, dim, training_length=data_length+num_latents+num_more_latents, dim=dim, num_layers=num_layers_encoder, heads=heads, mlp_mult=mlp_mult, pos_emb=pos_emb)
        self.frontend_pre_decoder_up = UpFrontend(frontend_encoder_layers_list).to(memory_format=torch.channels_last)

        self.lat2patch = init(nn.Linear(bottleneck_channels, dim))
        self.frontend_decoder_down = DownFrontend(frontend_layers_list, cond_dim=cond_channels).to(memory_format=torch.channels_last)
        if num_more_latents>0:
            self.more_latents_decoder = nn.Parameter(scale*torch.randn(1, num_more_latents, dim), requires_grad=True)
        else:
            self.more_latents_decoder = None
        self.decoder = Transformer_Diffusion(dim, dim, training_length=(data_length+num_latents+num_more_latents)*2, cond_dim=cond_channels, dim=dim, num_layers=num_layers, heads=heads, mlp_mult=mlp_mult, pos_emb=pos_emb)
        self.frontend_decoder_up = UpFrontend(frontend_layers_list, cond_dim=cond_channels).to(memory_format=torch.channels_last)

        self.fsq = FSQ(levels=fsq_levels)

    def get_attn_mask(self, x):
        """Generate attention mask for autoregressive decoding."""
        if x.shape[-1]==(2*spec_length):
            causal_mask = torch.triu(torch.ones(2, 2, device=x.device), diagonal=1)
            causal_mask = torch.nn.functional.interpolate(causal_mask.unsqueeze(0).unsqueeze(0), scale_factor=data_length+num_latents+num_more_latents, mode='nearest-exact').squeeze(0).squeeze(0).bool()
            pos_bias = torch.zeros_like(causal_mask).float()
            pos_bias.masked_fill_(causal_mask, -torch.finfo(x.dtype).max)
            return pos_bias
        else:
            raise ValueError(f'Invalid data length. Must be {(2*spec_length)} while it is {x.shape[-1]}.')

    def encoder_forward(self, x, dont_quantize=False, log_magnitude=False):
        """Encode STFT chunks to latents.

        Args:
            x: Input spectrogram chunks [B, C, F, T], T multiple of spec_length.
            dont_quantize: If True, returns continuous latents scaled to [-1,1].
            log_magnitude: If True, prints intermediate magnitudes (debug).
        Returns:
            Tensor of latents with shape [B, num_latents*(T/spec_length), dim].
        """
        assert x.shape[-1]%spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {spec_length}.'
        factor = None
        if x.shape[-1]>spec_length:
            x_ls = torch.split(x, spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        if dont_quantize:
            x = self.fsq.dont_quantize(x)
        else:
            x = self.fsq.quantize(x)
        if factor is not None:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
        return x
    
    @torch.compile(fullgraph=True, dynamic=False, mode='max-autotune-no-cudagraphs')
    def encoder_forward_fast(self, x, extract_features=False, dont_quantize=False, log_magnitude=False):
        """torch.compile-optimized variant of encoder_forward with same outputs."""
        assert x.shape[-1]%spec_length==0, f'Input shape {x.shape[-1]} is not divisible by {spec_length}.'
        factor = None
        if x.shape[-1]>spec_length:
            x_ls = torch.split(x, spec_length, dim=-1)
            factor = len(x_ls)
            x = torch.cat(x_ls, dim=0)
        x = self.frontend_encoder_down(x, gain=self.gain_encoder, log_magnitude=log_magnitude)[0]
        if self.more_latents_encoder is not None:
            x = self.encoder(x, torch.cat((self.latents.expand(x.shape[0], -1, -1), self.more_latents_encoder.expand(x.shape[0], -1, -1)), -2), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        else:
            x = self.encoder(x, self.latents.expand(x.shape[0], -1, -1), return_latents=True, skip_input_layer=True, skip_output_layer=False, print_magnitudes=log_magnitude)[:, :num_latents]
        if dont_quantize:
            x = self.fsq.dont_quantize(x)
        else:
            x = self.fsq.quantize(x)
        if factor is not None:
            x = torch.cat(torch.chunk(x, factor, dim=0), dim=-2)
        return x
    
    def pre_decoder_forward(self, latents, log_magnitude=False):
        """Project latents and generate multi-scale features for the decoder.

        Args:
            latents: [B, L, dim] where L is multiple of num_latents.
        Returns:
            List of feature maps aligned with decoder frontend stages.
        """
        assert latents.shape[-2]%num_latents==0, f'Input shape {latents.shape[-2]} is not divisible by {num_latents}.'
        factor = None
        if latents.shape[-2]>num_latents:
            latents_ls = torch.split(latents, num_latents, dim=-2)
            factor = len(latents_ls)
            latents = torch.cat(latents_ls, dim=0)
        mask_embedding = self.mask_embedding.expand(latents.shape[0], -1, self.time_dim, -1).reshape(latents.shape[0], -1, dim)
        if self.more_latents_pre_decoder is not None:
            x = self.pre_decoder(mask_embedding, torch.cat((self.lat2patch_pre_decoder(latents), self.more_latents_pre_decoder.expand(latents.shape[0], -1, -1)), -2), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        else:
            x = self.pre_decoder(mask_embedding, self.lat2patch_pre_decoder(latents), return_latents=False, skip_input_layer=True, skip_output_layer=True, print_magnitudes=log_magnitude)
        features = self.frontend_pre_decoder_up(x, skip_output_layer=True, log_magnitude=log_magnitude)[1]
        if factor is not None:
            features = [torch.cat(torch.chunk(el, factor, dim=0), dim=-1) for el in features]
        return features
    
    def decoder_forward(self, x, latents, features=None, sigma_left=None, sigma_right=None, output='both', log_magnitude=False):
        """Single denoising step conditioned on left/right sigmas.

        Inputs are split along time into left/right halves. Returns left, right,
        or both halves depending on `output`.
        """

        if sigma_left is None:
            if output=='left':
                sigma_left = sigma_max
            else:
                sigma_left = sigma_min
        if sigma_right is None:
            sigma_right = sigma_max
        
        # CONDITIONING
        sigma_left = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)*sigma_left
        sigma_right = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)*sigma_right
        sigma = torch.cat([sigma_left, sigma_right], dim=0)
        sigma_log = torch.log(sigma)/4.
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self.emb_proj(emb_sigma_log)

        gain = self.gain_decoder(emb_sigma_log).unsqueeze(-2).unsqueeze(-1)+1.
        gain_inp, gain_out = torch.chunk(gain, 2, dim=-2)

        if features is None:
            features = self.pre_decoder_forward(latents, log_magnitude=log_magnitude)
        features = [torch.chunk(el, 2, dim=-1) for el in features]
        features = [torch.cat(el, dim=0) for el in features]
            
        c_skip, c_out, c_in = get_c(sigma)
        attn_mask = self.get_attn_mask(x)
        x = torch.chunk(x, 2, dim=-1)
        x = torch.cat(x, dim=0)
        inp = x.clone()
        x = c_in*x
        x, features_dec = self.frontend_decoder_down(x, cond=time_emb, features=features, gain=gain_inp, log_magnitude=log_magnitude)
        x = torch.cat(torch.chunk(x, 2, dim=0), dim=-2)
        time_emb_left, time_emb_right = torch.chunk(time_emb, 2, dim=0)
        time_emb_transformer = torch.cat((torch.ones((x.shape[0], x.shape[1]//2 +num_latents+num_more_latents, time_emb.shape[-1]), dtype=x.dtype, device=x.device)*time_emb_left.unsqueeze(-2), torch.ones((x.shape[0], x.shape[1]//2 +num_latents +num_more_latents, time_emb.shape[-1]), dtype=x.dtype, device=x.device)*time_emb_right.unsqueeze(-2)), dim=-2)
        if self.more_latents_decoder is not None:
            x = self.decoder(x, time_emb_transformer, self.lat2patch(latents), self.more_latents_decoder.expand(x.shape[0], -1, -1), skip_input_layer=True, skip_output_layer=True, attn_mask=attn_mask, print_magnitudes=log_magnitude)
        else:
            x = self.decoder(x, time_emb_transformer, self.lat2patch(latents), skip_input_layer=True, skip_output_layer=True, attn_mask=attn_mask, print_magnitudes=log_magnitude)
        x = torch.cat(torch.chunk(x, 2, dim=-2), dim=0)
        x = self.frontend_decoder_up(x, cond=time_emb, features=features_dec, gain=gain_out, log_magnitude=log_magnitude)[0]
        x = c_skip*inp + c_out*x
        x_left, x_right = torch.chunk(x, 2, dim=0)
        if output=='left':
            return x_left
        elif output=='right':
            return x_right
        elif output=='both':
            return torch.cat((x_left, x_right), dim=-1)
        elif output=='features':
            return torch.cat((x_left, x_right), dim=-1), features
        else:
            raise ValueError(f'Invalid output type: {output}. Must be one of: (left, right, both, features).')

    def decode_single_parallel(self, latents, denoising_steps=None):
        """Run the full parallel denoising schedule for one sample/batch."""
        if denoising_steps is None:
            denoising_steps = default_denoising_steps_parallel
        step_size = 1./denoising_steps
        noise = torch.randn((latents.shape[0], stft_channels, hop*(fac//2), spec_length*(latents.shape[-2]//num_latents)), dtype=latents.dtype, device=latents.device)*sigma_max
        latents_ls = torch.split(latents, num_latents, dim=-2)
        num_chunks = len(latents_ls)
        features = self.pre_decoder_forward(torch.cat(latents_ls, dim=0))
        sigma = sigma_max
        inp = noise
        for step in range(denoising_steps):
            inp, num_samples = preprocess_parallel_input(inp, iteration=step, length=spec_length, dim=-1)
            inp_latents, num_samples = preprocess_parallel_input(latents, iteration=step, length=num_latents, dim=-2)
            inp_features = preprocess_parallel_features(features, iteration=step, num_samples=num_chunks, dim=-1)
            out = self.decoder_forward(inp, inp_latents, inp_features, output='both', sigma_left=sigma, sigma_right=sigma)
            out = postprocess_parallel_input(out, iteration=step, num_samples=num_samples, length=spec_length, dim=-1)
            sigma = get_sigma_continuous(1.-(step+1)*step_size)
            inp = reverse_step(out, torch.randn_like(out), sigma)
        return out
    
    def decode_parallel(self, latents, denoising_steps=None, max_batch_size=None):
        """Decode all timesteps in parallel using `distribute` for batching."""
        device = next(self.parameters()).device
        out = distribute(self.decode_single_parallel, latents, max_batch_size, device, denoising_steps=denoising_steps)
        return out

    def decode_autoregressive(self, latents, time_prompt=None, denoising_steps=None, max_batch_size=None):
        """Autoregressive decoding over timesteps."""
        if denoising_steps is None:
            denoising_steps = default_denoising_steps_ar
        if time_prompt is None:
            time_prompt = default_time_prompt
        step_size = 1./denoising_steps
        # step_size_prompt = time_prompt/denoising_steps
        device = next(self.parameters()).device
        sigma_prompt = get_sigma_continuous(time_prompt)
        out_ls = []
        for i in range(latents.shape[-2]//num_latents):
            if i==0:
                noise = torch.randn((latents.shape[0], stft_channels, hop*(fac//2), spec_length), dtype=latents.dtype, device=latents.device)*sigma_max
                sigma = sigma_max
                features = distribute(self.pre_decoder_forward, latents[:, :num_latents*2], max_batch_size, device)
                for step in range(denoising_steps):
                    out = distribute(self.decoder_forward, torch.cat((noise, torch.zeros_like(noise)), dim=-1), max_batch_size, device, latents[:, :num_latents*2], features=features, output='left', sigma_left=sigma)
                    sigma = get_sigma_continuous(1.-(step+1)*step_size)
                    noise = reverse_step(out, torch.randn_like(out), sigma)
                out_ls.append(out)
            else:
                last_out_clean = out_ls[-1]
                noise = torch.randn_like(last_out_clean)*sigma_max
                last_out = reverse_step(last_out_clean, torch.randn_like(last_out_clean), torch.ones_like(last_out_clean)*sigma_prompt)
                sigma = sigma_max
                sigma_last_out = sigma_prompt
                features = distribute(self.pre_decoder_forward, latents[:, (i-1)*num_latents:(i+1)*num_latents], max_batch_size, device)
                for step in range(denoising_steps):
                    out = distribute(self.decoder_forward, torch.cat([last_out, noise], dim=-1), max_batch_size, device, latents[:, (i-1)*num_latents:(i+1)*num_latents], features=features, output='right', sigma_left=sigma_last_out, sigma_right=sigma)
                    sigma = get_sigma_continuous(1.-(step+1)*step_size)
                    noise = reverse_step(out, torch.randn_like(out), sigma)
                out_ls.append(out)
        out = torch.cat(out_ls, dim=-1)
        return out
    
    def decode_autoregressive_step(self, current_latents, past_repr=None, past_latents=None, time_prompt=None, denoising_steps=None, max_batch_size=None):
        """Single-step live decoding with optional past spectrogram and latents."""
        if denoising_steps is None:
            denoising_steps = default_denoising_steps_ar
        if time_prompt is None:
            time_prompt = default_time_prompt
        step_size = 1./denoising_steps
        device = next(self.parameters()).device
        sigma_prompt = get_sigma_continuous(time_prompt)
        if past_repr is None:
            noise = torch.randn((current_latents.shape[0], stft_channels, hop*(fac//2), spec_length), dtype=current_latents.dtype, device=current_latents.device)*sigma_max
            sigma = sigma_max
            features = distribute(self.pre_decoder_forward, current_latents[:, -num_latents:], max_batch_size, device)
            features = [torch.cat((el, torch.zeros_like(el)), dim=-1) for el in features]
            for step in range(denoising_steps):
                out = distribute(self.decoder_forward, torch.cat((noise, torch.zeros_like(noise)), dim=-1), max_batch_size, device, torch.cat((current_latents[:, -num_latents:], torch.zeros_like(current_latents[:, -num_latents:])), dim=-2), features=features, output='left', sigma_left=sigma)
                sigma = get_sigma_continuous(1.-(step+1)*step_size)
                noise = reverse_step(out, torch.randn_like(out), sigma)
            out = torch.cat((torch.zeros_like(out), out), dim=-1)
        else:
            last_out_clean = past_repr[..., -spec_length:]
            if past_latents is None:
                past_latents = distribute(self.encoder_forward, last_out_clean, max_batch_size, device)
            else:
                past_latents = past_latents[:, -num_latents:]
            noise = torch.randn_like(last_out_clean)*sigma_max
            last_out = reverse_step(last_out_clean, torch.randn_like(last_out_clean), torch.ones_like(last_out_clean)*sigma_prompt)
            sigma = sigma_max
            sigma_last_out = sigma_prompt
            features = distribute(self.pre_decoder_forward, torch.cat((past_latents, current_latents[:, -num_latents:]), dim=-2), max_batch_size, device)
            for step in range(denoising_steps):
                out = distribute(self.decoder_forward, torch.cat([last_out, noise], dim=-1), max_batch_size, device, torch.cat((past_latents, current_latents[:, -num_latents:]), dim=-2), features=features, output='right', sigma_left=sigma_last_out, sigma_right=sigma)
                sigma = get_sigma_continuous(1.-(step+1)*step_size)
                noise = reverse_step(out, torch.randn_like(out), sigma)
            out = torch.cat((last_out_clean, out), dim=-1)
        return out