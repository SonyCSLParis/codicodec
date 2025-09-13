import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F
import einops

from .hparams import *
from .hparams_inference import *
from .utils import is_path, distribute, is_integer, download_model
from .models import UNet
from .audio import to_representation_encoder, to_waveform


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if mixed_precision:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class EncoderDecoder:
    """Codec wrapper for encoding waveforms to latents and decoding them back.

    Handles model loading, device placement, and batching utilities. Public API
    mirrors previous releases while using the new codicodec architecture.
    """
    def __init__(self, load_path_inference=None, device=None):
        # Ensure checkpoint is available locally (downloaded from HF Hub if needed)
        download_model()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.load_path_inference = load_path_inference
        if load_path_inference is None:
            self.load_path_inference = load_path_inference_default
        self.get_models()
        self.latents_per_timestep = num_latents
        self.bottleneck_channels = bottleneck_channels

        # live decoding
        self.past_spec = None
        self.past_latents = None

    def latents2dim(self, latents, desired_channels=64):
        """Reshape latent channels to desired size while preserving content."""
        assert desired_channels%self.bottleneck_channels==0, f"Desired channels must be divisible by original number of channels = {self.bottleneck_channels}"
        return einops.rearrange(latents, '... (l d) c -> ... l (d c)', d=desired_channels//self.bottleneck_channels)
    
    def dim2latents(self, latents):
        """Inverse of latents2dim()."""
        return einops.rearrange(latents, '... l (d c) -> ... (l d) c', c=self.bottleneck_channels)
        
    def get_models(self):
        """Instantiate UNet and load weights from checkpoint."""
        gen = UNet().to(self.device)
        gen.eval()
        checkpoint = torch.load(self.load_path_inference, map_location=self.device, weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=True)
        self.gen = gen

    def encode(self, path_or_audio, max_batch_size=None, discrete=False, preprocess_on_gpu=True, desired_channels=64, fix_batch_size=False):
        '''
        path_or_audio: path of audio sample to encode or numpy array of waveform to encode
        max_batch_size: maximum inference batch size for encoding: tune it depending on the available GPU memory

        WARNING! if input is numpy array of stereo waveform, it must have shape [audio_channels, waveform_samples]

        Returns latents with shape [audio_channels, dim, length]
        '''
        if max_batch_size is None:
            max_batch_size = max_batch_size_encode
        if discrete:
            # For discrete encoding, always quantize before extracting codebook indices
            latents = encode_audio_inference(path_or_audio, self, max_batch_size, device=self.device, dont_quantize=False, preprocess_on_gpu=preprocess_on_gpu, fix_batch_size=fix_batch_size)
            return self.gen.fsq.codes_to_indexes(latents)
        # Continuous or quantized continuous encoding
        latents = encode_audio_inference(path_or_audio, self, max_batch_size, device=self.device, dont_quantize=True, preprocess_on_gpu=preprocess_on_gpu, fix_batch_size=fix_batch_size)
        # reshape to desired channels
        out = self.latents2dim(latents, desired_channels=desired_channels)
        # apply inverse tanh transform and rescale for continuous latents
        out = torch.atanh(out) / sigma_rescale
        return out
    
    def decode(self, latent, mode='parallel', max_batch_size=None, denoising_steps=None, time_prompt=None, preprocess_on_gpu=True):
        '''
        latent: numpy array of latents to decode with shape [audio_channels, dim, length]
        max_batch_size: maximum inference batch size for decoding: tune it depending on the available GPU memory
        time_prompt: noise level added to past token

        Returns numpy array of decoded waveform with shape [waveform_samples, audio_channels]
        '''
        # if dtype of latents is int32 or int64, then set discrete to True
        discrete = is_integer(latent)
        if max_batch_size is None: 
            max_batch_size = max_batch_size_decode
        if discrete:
            latents = self.gen.fsq.indexes_to_codes(latent)
        else:
            # invert rescaling and transform for continuous latents
            inv = latent * sigma_rescale
            inv = torch.tanh(inv)
            latents = self.dim2latents(inv)
        return decode_latent_inference(latents, self, mode, max_batch_size, denoising_steps=denoising_steps, time_prompt=time_prompt, device=self.device, preprocess_on_gpu=preprocess_on_gpu)
    
    def reset(self):
        """Clear internal live-decoding buffers."""
        self.past_spec = None
        self.past_latents = None

    def decode_next(self, latents, max_batch_size=None, denoising_steps=None, discrete=False, time_prompt=None, preprocess_on_gpu=True):
        '''
        latents: numpy array of latents to decode with shape [audio_channels, dim, length]
        max_batch_size: maximum inference batch size for decoding: tune it depending on the available GPU memory
        time_prompt: noise level added to past token

        Returns numpy array of decoded waveform with shape [waveform_samples, audio_channels]
        '''
        if max_batch_size is None: 
            max_batch_size = max_batch_size_decode
        if discrete:
            latents = self.gen.fsq.indexes_to_codes(latents)
        else:
            # invert rescaling and transform for continuous latents
            inv = latents * sigma_rescale
            inv = torch.tanh(inv)
            latents = self.dim2latents(inv)
        wv, past_spec, past_latents = decode_next_latent_inference(latents, self, max_batch_size, denoising_steps=denoising_steps, time_prompt=time_prompt, device=self.device, preprocess_on_gpu=preprocess_on_gpu)
        self.past_spec = past_spec
        self.past_latents = past_latents
        return wv






# Encode audio sample for inference
# Parameters:
#   audio_path: path of audio sample
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   latent: compressed latent representation with shape [audio_channels, latent_length, dim]
@torch.no_grad()
def encode_audio_inference(audio_path, trainer, max_batch_size_encode, device='cuda', dont_quantize=False, preprocess_on_gpu=False, fix_batch_size=False):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    squeeze_batch_dimensions = False
    if is_path(audio_path):
        audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = np.transpose(audio, [1,0])
    else:
        audio = audio_path
        sr = None
        if len(audio.shape)==1:
            squeeze_batch_dimensions = True
            # check if audio is numpy array, then use np.expand_dims, if it is a pytorch tensor, then use torch.unsqueeze
            if isinstance(audio, np.ndarray):
                audio = np.expand_dims(audio, 0)
                if stereo:
                    audio = np.repeat(audio, 2, axis=0)
            else:
                audio = torch.unsqueeze(audio, 0)
                if stereo:
                    audio = torch.repeat_interleave(audio, 2, dim=0)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if preprocess_on_gpu:
        audio = audio.to(device)
    else:
        audio = audio.cpu()
    audio_channels = audio.shape[-2]
    if audio_channels==1 and stereo:
        audio = torch.cat([audio, audio], -2)

    if stereo and len(audio.shape)==2:
        squeeze_batch_dimensions = True
        audio = torch.unsqueeze(audio, 0)
    if len(audio.shape)>3:
        raise ValueError("Input audio shape is not valid. It should be [waveform_samples], [audio_channels, waveform_samples] or [batch_size, audio_channels, waveform_samples]")

    batch_size = audio.shape[0]
    repr_encoder = to_representation_encoder(audio)
    del audio

    if repr_encoder.shape[-1]%spec_length!=0:
        pad = spec_length-(repr_encoder.shape[-1]%spec_length)
        repr_encoder = F.pad(repr_encoder, (0,pad))

    if repr_encoder.shape[-1]>spec_length:
        repr_encoder = torch.split(repr_encoder, spec_length, dim=-1)
        repr_encoder = torch.cat(repr_encoder, dim=0)

    device = next(trainer.gen.parameters()).device
    if fix_batch_size:
        original_batch_size = repr_encoder.shape[0]
        # make sure that batch size is exactly divisible by max_batch_size_encode
        if repr_encoder.shape[0]%max_batch_size_encode!=0:
            rem = torch.zeros(max_batch_size_encode-(repr_encoder.shape[0]%max_batch_size_encode), *repr_encoder.shape[1:], device=repr_encoder.device, dtype=repr_encoder.dtype)
            repr_encoder = torch.cat([repr_encoder, rem], 0)
        latent = distribute(trainer.gen.encoder_forward_fast, repr_encoder, max_batch_size_encode, device, dont_quantize=dont_quantize)
        latent = latent[:original_batch_size]
    else:
        latent = distribute(trainer.gen.encoder_forward, repr_encoder, max_batch_size_encode, device, dont_quantize=dont_quantize)
        
    del repr_encoder
    # split samples
    latent = torch.split(latent, batch_size, 0)
    latent = torch.stack(latent, -3)
    if latent.shape[0]==1 and squeeze_batch_dimensions:
        latent = latent.squeeze(0)
    return latent



# Decode latent representation for inference, use the same framework as in encode_audio_inference, but in reverse order for decoding
# Parameters:
#   latent: compressed latent representation with shape [batch_size, timesteps, latents_per_timestep, dim] or [timesteps, latents_per_timestep, dim]
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   audio: numpy array of decoded waveform with shape [waveform_samples, audio_channels]
@torch.no_grad()
def decode_latent_inference(latent, trainer, mode, max_batch_size_decode, denoising_steps=None, device='cuda', preprocess_on_gpu=False, time_prompt=None):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    # check if latent is numpy array, then convert to tensor
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)
    if preprocess_on_gpu:
        latent = latent.to(device)
    else:
        latent = latent.cpu()
    squeeze_batch_dimensions = False
    # if latent has only 3 dimensions, add a third dimension as axis 0
    if len(latent.shape)==3:
        squeeze_batch_dimensions = True
        latent = torch.unsqueeze(latent, 0)
    latent = torch.cat(torch.unbind(latent, -3), -2)
    original_length = (latent.shape[-2]//num_latents)*spec_length
    if mode=='parallel':
        # pad latents
        if latent.shape[-2]%(num_latents*2)!=0:
            pad = (num_latents*2)-(latent.shape[-2]%(num_latents*2))
            latent = F.pad(latent, (0,0,0,pad))

    if mode=='parallel':
        repr = trainer.gen.decode_parallel(latent, denoising_steps=denoising_steps, max_batch_size=max_batch_size_decode)
    elif mode=='autoregressive':
        repr = trainer.gen.decode_autoregressive(latent, time_prompt=time_prompt, denoising_steps=denoising_steps, max_batch_size=max_batch_size_decode)
    else:
        raise ValueError(f"Mode must be either 'parallel' or 'autoregressive', but got {mode}")
    if not preprocess_on_gpu:
        repr = repr.cpu()
    repr = to_waveform(repr[..., :original_length]).cpu()
    del latent
    if squeeze_batch_dimensions:
        repr = repr.squeeze(0)
    return repr


# Decode next latent representation for inference
# Parameters:
#   latent: compressed latent representation with shape [batch_size, 1, latents_per_timestep, dim] or [1, latents_per_timestep, dim] or [latents_per_timestep, dim]
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   audio: numpy array of decoded waveform with shape [waveform_samples, audio_channels]
@torch.no_grad()
def decode_next_latent_inference(latent, trainer, max_batch_size_decode, denoising_steps=None, device='cuda', preprocess_on_gpu=False, time_prompt=None):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    # check if latent is numpy array, then convert to tensor
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)
    if preprocess_on_gpu:
        latent = latent.to(device)
    else:
        latent = latent.cpu()
    squeeze_batch_dimensions = False
    # if latents has only 2 dimensions, add a first dimension as axis 0
    if len(latent.shape)==2:
        latent = torch.unsqueeze(latent, 0)
    # if latent has only 3 dimensions, add a third dimension as axis 0
    if len(latent.shape)==3:
        squeeze_batch_dimensions = True
        latent = torch.unsqueeze(latent, 0)
    latent = torch.squeeze(latent, -3)
    # equivalent way of doing the above line in a more elegant way:
    # latent = latent.permute(0, 2, 1, 3).reshape(latent.shape[0], -1, latent.shape[-1])

    repr = trainer.gen.decode_autoregressive_step(latent, past_repr=trainer.past_spec, past_latents=trainer.past_latents, time_prompt=time_prompt, denoising_steps=denoising_steps, max_batch_size=max_batch_size_decode)
    past_spec = repr[..., -spec_length:]
    if not preprocess_on_gpu:
        repr = repr.cpu()
    repr = to_waveform(repr).cpu()
    repr = repr[..., spec_length*hop:-(fac-1)*hop]
    if squeeze_batch_dimensions:
        repr = repr.squeeze(0)
    return repr, past_spec, latent