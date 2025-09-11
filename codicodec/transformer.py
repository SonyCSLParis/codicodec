import torch
from torch import nn

from .transformer_layers import MLP, AttentionBlock, AdaptiveNorm, zero_init, init


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, training_length=None, dim=512, num_layers=12, heads=8, mlp_mult=4, pos_emb='learned', autoregressive=False, latents_per_timestep=None, dropout=0.):
        super().__init__()

        self.training_length = training_length
        self.autoregressive = autoregressive
        scale = float(dim) ** -0.5

        self.pe = None
        if 'learned' in pos_emb:
            assert training_length is not None, "Training length must be provided for learned positional embeddings"
            self.pe = nn.Parameter(scale*torch.randn((1, training_length, dim)), requires_grad=True)

        self.pe_latents_per_timestep = None
        if latents_per_timestep is not None:
            self.pe_latents_per_timestep = nn.Parameter(scale*torch.randn((1, latents_per_timestep, dim)), requires_grad=True)

        self.linear_input = init(nn.Linear(input_dim, dim))
        self.norm_input = AdaptiveNorm(dim)
        self.norm_output = AdaptiveNorm(dim)
        self.linear_output = zero_init(nn.Linear(dim, output_dim))
        
        self.layers = nn.ModuleList([
            AttentionBlock(dim, heads=heads, mlp_mult=mlp_mult, training_length=training_length, causal=self.autoregressive, pos_emb=pos_emb, group_together=latents_per_timestep, dropout=dropout) for _ in range(num_layers) 
        ])        
        
    def forward(self, x, latent, return_latents=False, skip_input_layer=False, skip_output_layer=False, attn_mask=None, print_magnitudes=False):
        # x: samples with shape (batch_size, length, channels)

        if not skip_input_layer:
            x = self.linear_input(x)
            if print_magnitudes:
                print(f"Linear Input: {x.abs().mean()}")

        x = torch.cat([x, latent], dim=-2)

        if self.pe is not None:
            pe = self.pe[:, :x.size(-2)]
            x = x + pe

        if self.pe_latents_per_timestep is not None:
            pe_latents_per_timestep = self.pe_latents_per_timestep.repeat(1, x.size(-2)//self.pe_latents_per_timestep.size(-2), 1)
            x = x + pe_latents_per_timestep

        x = self.norm_input(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x, attn_mask=attn_mask)
            if print_magnitudes:
                print(f"Layer {i}: {x.abs().mean()}")

        if return_latents:
            x = x[:, -latent.size(-2):]
        else:
            x = x[:, :-latent.size(-2)]

        if not skip_output_layer:
            x = self.norm_output(x)
            x = self.linear_output(x)
            if print_magnitudes:
                print(f"Linear Output: {x.abs().mean()}")

        return x


class Transformer_Diffusion(nn.Module):
    def __init__(self, input_dim, output_dim, training_length=None, cond_dim=None, dim=512, num_layers=12, heads=8, mlp_mult=4, pos_emb='learned', autoregressive=False, latents_per_timestep=None, dropout=0.):
        super().__init__()

        self.training_length = training_length
        self.autoregressive = autoregressive
        scale = float(dim) ** -0.5
        
        self.pe = None
        if 'learned' in pos_emb:
            assert training_length is not None, "Training length must be provided for learned positional embeddings"
            self.pe = nn.Parameter(scale*torch.randn((1, training_length, dim)), requires_grad=True)

        self.pe_latents_per_timestep = None
        if latents_per_timestep is not None:
            self.pe_latents_per_timestep = nn.Parameter(scale*torch.randn((1, latents_per_timestep, dim)), requires_grad=True)

        self.linear_input = init(nn.Linear(input_dim, dim))
        self.norm_input = AdaptiveNorm(dim, cond_dim=cond_dim)
        self.norm_output = AdaptiveNorm(dim, cond_dim=cond_dim)
        self.linear_output = zero_init(nn.Linear(dim, output_dim))

        if cond_dim is None:
            raise ValueError("Dimensionality of conditioning cond_dim must be provided!")
        
        self.layers = nn.ModuleList([
            AttentionBlock(dim, heads=heads, mlp_mult=mlp_mult, cond_dim=cond_dim, training_length=training_length, causal=self.autoregressive, pos_emb=pos_emb, group_together=latents_per_timestep, dropout=dropout) for _ in range(num_layers) 
        ])

    def forward(self, x, cond, latent, more_latent=None, skip_input_layer=False, skip_output_layer=False, attn_mask=None, print_magnitudes=False):
        # x: noisy samples with shape (batch_size, length, channels)
        # cond: conditioning information with shape (batch_size, length, channels). 
        #       cond is usually the ouput of a MLP that takes as input the time embedding (and class information if applicable)

        if not skip_input_layer:
            x = self.linear_input(x)
            if print_magnitudes:
                print(f"Linear Input: {x.abs().mean()}")

        x1,x2 = torch.chunk(x, chunks=2, dim=-2)
        lat1,lat2 = torch.chunk(latent, chunks=2, dim=-2)
        if more_latent is not None:
            x = torch.cat([x1, lat1, more_latent, x2, lat2, more_latent], dim=-2)
        else:
            x = torch.cat([x1, lat1, x2, lat2], dim=-2)

        if self.pe is not None:
            pe = self.pe[:, :x.size(-2)]
            x = x + pe

        if self.pe_latents_per_timestep is not None:
            pe_latents_per_timestep = self.pe_latents_per_timestep.repeat(1, x.size(-2)//self.pe_latents_per_timestep.size(-2), 1)
            x = x + pe_latents_per_timestep

        x = self.norm_input(x, cond)

        for i in range(len(self.layers)):
            x = self.layers[i](x, cond, attn_mask=attn_mask)
            if print_magnitudes:
                print(f"Layer {i}: {x.abs().mean()}")

        x = self.norm_output(x, cond)

        xlat1, xlat2 = torch.chunk(x, chunks=2, dim=-2)
        x1 = xlat1[:, :x1.size(-2)]
        x2 = xlat2[:, :x2.size(-2)]
        x = torch.cat([x1, x2], dim=-2)

        if not skip_output_layer:
            x = self.linear_output(x)
            if print_magnitudes:
                print(f"Linear Output: {x.abs().mean()}")

        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, dim=512, num_layers=12, mlp_mult=1, dropout=0.):
        super().__init__()

        self.linear_input = init(nn.Linear(input_dim, dim))
        self.norm_output = AdaptiveNorm(dim)
        self.linear_output = zero_init(nn.Linear(dim, output_dim))
        
        self.layers = nn.ModuleList([
            MLP(dim, mlp_mult=mlp_mult, dropout=dropout) for _ in range(num_layers) 
        ])

    def forward(self, x, print_magnitudes=False):
        # x: samples with shape (batch_size, ..., channels)

        x = self.linear_input(x)
        if print_magnitudes:
            print(f"Linear Input: {x.abs().mean()}")

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if print_magnitudes:
                print(f"Layer {i}: {x.abs().mean()}")

        # get output
        x = self.norm_output(x)
        x = self.linear_output(x)
        if print_magnitudes:
            print(f"Linear Output: {x.abs().mean()}")

        return x


class MultiLayerPerceptron_Diffusion(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim=None, dim=512, num_layers=12, mlp_mult=1, dropout=0.):
        super().__init__()
        
        self.linear_input = init(nn.Linear(input_dim, dim))
        self.norm_output = AdaptiveNorm(dim, cond_dim=cond_dim)
        self.linear_output = zero_init(nn.Linear(dim, output_dim))
        
        self.layers = nn.ModuleList([
            MLP(dim, mlp_mult=mlp_mult, cond_dim=cond_dim, dropout=dropout) for _ in range(num_layers) 
        ])

    def forward(self, x, cond, print_magnitudes=False):
        # x: samples with shape (batch_size, ..., channels)

        x = self.linear_input(x)
        if print_magnitudes:
            print(f"Linear Input: {x.abs().mean()}")

        for i in range(len(self.layers)):
            x = self.layers[i](x, cond)
            if print_magnitudes:
                print(f"Layer {i}: {x.abs().mean()}")

        # get output
        x = self.norm_output(x, cond)
        x = self.linear_output(x)
        if print_magnitudes:
            print(f"Linear Output: {x.abs().mean()}")

        return x
