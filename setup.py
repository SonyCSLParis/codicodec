from setuptools import setup,find_packages

setup(
    name='codicodec',
    version='0.1.0',
    packages=find_packages(),
    description='Encode and decode audio samples to/from continuous and discrete compressed representations!',
    author='Sony Computer Science Laboratories Paris',
    author_email='music@csl.sony.fr', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    install_requires=[
        'numpy',
        'einops',
        'soundfile',
        'huggingface_hub',
        'librosa',
        'torch',
    ],
    license='CC BY-NC 4.0',
    url='https://github.com/SonyCSLParis/codicodec',
    keywords='audio speech music codec compression generative-model autoencoder diffusion continuous discrete latent'
)