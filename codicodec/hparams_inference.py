import os

filepath = os.path.abspath(__file__)
lib_root = os.path.dirname(filepath)

load_path_inference_default = os.path.join(lib_root, 'models/codicodec.pt')

max_batch_size_encode = 64                            # maximum inference batch size for encoding: tune it depending on the available GPU memory  
max_batch_size_decode = 32                            # maximum inference batch size for decoding: tune it depending on the available GPU memory

sigma_rescale = 0.8  # scaling constant for latent rescaling