# torch compile path should be the path where this file is located
import os
torch_compile_cache_dir = os.path.dirname(os.path.abspath(__file__)) + '/torch_compile_cache'

# GENERAL/INFERENCE
mixed_precision = True                                                      # use mixed precision
seed = 42                                                                   # seed for Pytorch

stereo = True                                                               # if True, train on stereo data, if False, train on mono data

default_time_prompt = 0.4                                                   # default time prompt for inference
default_denoising_steps_parallel = 5                                        # default number of denoising steps for inference
default_denoising_steps_ar = 2                                              # default number of denoising steps for inference

# stft spectrogram params
hop = 512*2
fac = 4//2
if stereo:
    stft_channels = 4
else:
    stft_channels = 2

sample_rate = 48000                                                         # sampling rate of input/output audio

# STFT normalization params
alpha_rescale = 0.65
beta_rescale = 0.34


# MODEL
dim = 512                                                                   # hidden transformer dimension
head_dim = 128                                                              # hidden dimension of each head in transformer
heads = dim//head_dim                                                       # number of heads in transformer
mlp_mult = 4                                                                # multiplier for hidden layer in transformer
pos_emb = 'learned'                                                         # if True, use positional embedding in transformer (alibi)
num_layers = 12                                                             # number of layers in diffusion backbone
num_layers_encoder = num_layers                                             # number of layers in encoder
cond_channels = 512                                                         # dimension of time embedding

num_latents = 128                                                           # number of latents per patch of data_length//2 tokens
num_more_latents = 8                                                        # number of additional latents per patch of data_length//2 tokens, to be discarded
fsq_levels = [11, 11, 11, 11]
bottleneck_channels = len(fsq_levels)   

# frontend params
frontend_base_channels = 64
frontend_multipliers_list = [1, 2, 4, dim//frontend_base_channels]
frontend_layers_list = [3, 3, 3, 1]
frontend_encoder_layers_list = frontend_layers_list
frontend_freq_downsample_list = [0, 1, 0]

spec_length = 32
downsample_ratio = (4**frontend_freq_downsample_list.count(0))*(4**frontend_freq_downsample_list.count(1))*(2**frontend_freq_downsample_list.count(2))*(2**frontend_freq_downsample_list.count(3))
data_length = (hop*(fac//2)*spec_length)//downsample_ratio                  # sequence length of data used for training

sigma_min = 0.002                                                           # minimum sigma
sigma_max = 80.                                                             # maximum sigma
sigma_data = 0.5                                                            # sigma for data
rho = 7.                                                                    # rho parameter for sigma schedule