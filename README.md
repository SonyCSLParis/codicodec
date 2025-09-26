![CoDiCodec Banner](codicodec_banner.jpg)

# CoDiCodec

**CoDiCodec** is a generative neural audio **Codec** that highly compresses 44.1/48 kHz stereo audio into both **Co**ntinuous and **Di**screte representations with SOTA reconstruction quality (as of Sep. 2025):
*   **Discrete tokens**: at a bitrate of 2.38 kbit/s.
*   **Continuous latent vectors**: at a ~11 Hz frame rate with 64 channels (128x compression).

## Installation

1.  Install via pip:
    ```bash
    pip install codicodec
    ```

## Quick Start: Encoding and Decoding

Here is a basic example:

```python
import librosa
import numpy as np
import IPython
from codicodec import EncoderDecoder

# Initialize the EncoderDecoder
encdec = EncoderDecoder()

# Load an example audio file (or use your own!)
audio_path = librosa.example('trumpet')
wv, sr = librosa.load(audio_path, sr=44100)  # Ensure 44.1kHz sample rate

# Encode the audio
latent = encdec.encode(wv)

# Decode the latent representation back to a waveform
wv_rec = encdec.decode(latent)

# Listen to the original and reconstructed audio
print('Original')
IPython.display.display(IPython.display.Audio(wv, rate=sr))
print('Reconstructed')
IPython.display.display(IPython.display.Audio(wv_rec.squeeze().cpu().numpy(), rate=sr))
```

## Understanding the `encode()` Function

The `encode()` function returns compressed representations.  Here is a breakdown its arguments:

*   **`path_or_audio`:**  This can be either:
    *   A string: The path to your audio file.
    *   A NumPy array or PyTorch tensor:  The waveform data itself.  The shape should be `[audio_channels, waveform_samples]` or `[waveform_samples]` (for mono).  If it's stereo and you provide a 1D array, it's automatically expanded to `[1, waveform_samples]` and duplicated for both channels.

*   **`max_batch_size`:**  This controls the maximum number of audio chunks processed in parallel during encoding.  Tune this based on your available GPU memory.  A larger `max_batch_size` generally leads to faster encoding, up to the limit of your GPU's capacity.  The default is set in `hparams_inference.py`.

*   **`discrete`:**
    *   `discrete=False` (default): The encoder returns the continuous latent vectors.
    *   `discrete=True`:  The encoder returns *integer indices* representing the quantized latent codes.  This is essential for training a language model on the compressed representations, as language models typically work with discrete tokens.  The indices correspond to entries in the FSQ codebook.

*   **`preprocess_on_gpu`:**
    *   `preprocess_on_gpu=True` (default):  The computationally intensive Short-Time Fourier Transform (STFT) operation is performed on the GPU.  This is significantly faster but requires more GPU memory.
    *   `preprocess_on_gpu=False`:  The STFT is done on the CPU. Use this if you run into GPU memory issues, especially with very long audio files.

*   **`desired_channels`:** This allows you to reshape the latent representation.  The default (64) results in a `[timesteps, latents_per_timestep, 64]` output.  You can change this to trade off between the number of channels and the length of the latent sequence.  For instance:
    *   `desired_channels=32`:  Produces `[timesteps, latents_per_timestep*2, 32]`, halving the channel dimension and doubling the `latents_per_timestep` dimension.

*   **`fix_batch_size`:**
    *   `fix_batch_size=False` (default): The encoder processes the audio in batches of varying sizes, up to `max_batch_size`.
    *   `fix_batch_size=True`:  The encoder *pads* the input to ensure that each batch has a size *exactly* equal to `max_batch_size`.  This is crucial for enabling `torch.compile`, which can dramatically speed up encoding.  `torch.compile` optimizes the computation graph, but it requires fixed input sizes.  The first time a new batch size is encountered, compilation will take some time, but subsequent runs with the same size will be much faster due to cached, optimized Triton kernels.  This is highly recommended for large-scale encoding tasks.

**Example: Discrete Latents for Language Modeling**

```python
latent_indices = encdec.encode(wv, discrete=True)
print(f"Shape of latent indices: {latent_indices.shape}")
print(f"Data type: {latent_indices.dtype}")  # Will be int64
print(latent_indices[0]) # Shows the first set of latent indices

```

**Example:  Controlling Latent Shape and Using `torch.compile`**

```python
# Encode with a different channel dimension and fixed batch size
latent_long = encdec.encode(wv, desired_channels=32, fix_batch_size=True, max_batch_size=32)
print(f"Shape of reshaped latents: {latent_long.shape}")

# The first run with fix_batch_size=True and a new max_batch_size will trigger compilation.
# Subsequent runs with the *same* max_batch_size will be much faster.
```

## Understanding the `decode()` Function

The `decode()` function transforms latent representations back into audio waveforms.

*   **`latent`:** A NumPy array or PyTorch tensor of latent embeddings.  The expected shape is `[audio_channels, dim, length]` (or `[batch_size, audio_channels, dim, length]`). If the dtype of the latents is integer (int32 or int64), `decode` automatically assumes they are discrete indices.

*   **`mode`:**
    *   `mode='parallel'` (default): Decodes the entire latent sequence in parallel. This is generally faster for offline processing.
    *   `mode='autoregressive'` : Decodes the sequence step-by-step, using past decoded output to inform the next step.  This is useful for generating longer sequences or for simulating a streaming scenario.
*    **`max_batch_size`:** Similar to `encode()`, this controls the maximum batch size for decoding.
*    **`denoising_steps`**: Number of denoising steps the model takes. It uses the default value specified in `hparams_inference.py` if no argument is supplied.
*    **`time_prompt`:** Level of noise that is added to past token when doing autoregressive decoding. It uses the default value specified in `hparams_inference.py` if no argument is supplied.
*   **`preprocess_on_gpu`:**
      *   `preprocess_on_gpu=True` (default): The inverse STFT operation is performed on the GPU.  This is faster but requires more GPU memory.
      *   `preprocess_on_gpu=False`: The inverse STFT is done on the CPU.  Use this if you encounter GPU memory issues.

**Example: Autoregressive Decoding**
```python
wv_rec_ar = encdec.decode(latent, mode='autoregressive')
IPython.display.display(IPython.display.Audio(wv_rec_ar.squeeze().cpu().numpy(), rate=sr))
```

## Live Decoding with `decode_next()` and `reset()`

For real-time applications, CoDiCodec provides `decode_next()` for live, streaming decoding:

```python
wv_ls = []
for i in range(latent.shape[0]):
    wv_chunk = encdec.decode_next(latent[i])
    wv_ls.append(wv_chunk)
wv_rec_live = torch.cat(wv_ls, axis=-1)
IPython.display.display(IPython.display.Audio(wv_rec_live.squeeze().cpu().numpy(), rate=sr))
# Reset the internal buffer before starting a new stream
encdec.reset()
```

*   **`decode_next(latents)`:**  Decodes the *next* segment of audio, given the current latent chunk (`latents`).  It internally maintains a buffer (`past_spec`, `past_latents`) of the previously decoded audio to ensure smooth transitions between chunks.  The output `wv_chunk` is a waveform segment ready to be played.

*   **`reset()`:**  Clears the internal buffer.  Call this *before* starting a new live decoding sequence.  It's essential to call `reset()` if you're switching to a new audio stream.

**Advantages of Live Decoding:**

*   **Low Latency:**  You get audio output as soon as each latent chunk is available.
*   **Real-time Applications:**  Ideal for interactive music systems and other scenarios where immediate decoded audio is needed.

## Summary and Best Practices

*   **Choose `discrete=False` for maximum reconstruction quality (this is the default behavior).**
*   **Use `discrete=True` when extracting data for language models.**
*   **Experiment with `max_batch_size` to optimize encoding/decoding speed.**
*   **Leverage `fix_batch_size=True` and `torch.compile` for significant speedups.**
*   **Use `preprocess_on_gpu=True` unless you encounter memory limitations.**
*   **For live applications, use `decode_next()` and `reset()` for low-latency, seamless decoding.**
* **The output shape of the encode function is [timesteps, latents_per_timestep, dim] or [batch_size, timesteps, latents_per_timestep, dim] if using batched inputs. When using transformer models you can also concatenate all latents along the same time axis, since transformers do not require an ordered sequence. However, it is recommended to use a learned positional embedding for each latent of the timestep.**

## Summary Embeddings

Unlike traditional methods that produce ordered sequences of latents, CoDiCodec's encoder generates a set of latents for each input audio chunk, each of which can encode global features.  This allows for significantly higher compression!

If your input waveform has shape `[audio_channels=2, waveform_samples]`, the encoder outputs a tensor of shape `[timesteps, latents_per_timestep, dim]`. You can then reshape the latents to `[timesteps*latents_per_timestep, dim]` to feed them into a transformer model. If you require a temporally ordered sequence of latents, you can reshape them to `[timesteps, latents_per_timestep*dim]` instead (this may be useful in case you do not use a permutation invariant model, such as a CNN), although the high channel dimension may lead to a higher computational cost in your downstream model.

![Architecture Diagram](architecture.png)

## License
This library is released under the CC BY-NC 4.0 license. Please refer to the LICENSE file for more details.



This work was conducted by [Marco Pasini](https://twitter.com/marco_ppasini) during his PhD at Queen Mary University of London, in partnership with Sony Computer Science Laboratories Paris.
This work was supervised by Stefan Lattner and George Fazekas.
