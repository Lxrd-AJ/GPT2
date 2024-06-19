from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads but batched
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # Project q,k,v in one step
        # output projection of the attention layer
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not a real bias term but name chosen to match OpenAI/HF? naming??
        # buffers are equivalent to States in MATLAB's custom layers
        # `bias` is a (1, 1, T, T) tensor, useful for broadcasting
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x) # computes the query, key & value in parallel, resulting tensor is (B,T,C) where C = 3 * n_embed
        q, k, v = qkv.split(self.n_embed, dim=2) # Split the BxTx(3*C) matrix into (B,T,C) chunks where C = n_embed

        # `nh` means number of heads
        # `hs` means head size for each head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        
        # attention (materialises the large (T,T) matrix for all the queries and keys)
        # Original Attention:
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Flash Attention:
        # This is not optimised for ANE as at of 2024, WWDC just happened so it's all beta, and alternative is to re-write the attention
        # following Apple's guide at https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 


        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C) where C = nh * hs
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length, `T`
    vocab_size: int = 50257 # token count: 50,000 BPE merges + 256 bytes tokens + 1 end of text token
    n_layer: int = 12 
    n_head: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([
                Block(config) for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight sharing/tying
        self.transformer.wte.weight = self.lm_head.weight

        # init params - `apply` is called for every module
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb

        # forward through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # forward through the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # Load pretrained GPT-2 model weights from hugging face
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),  # 774M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask
        
        # init a huggingface transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf  if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf  if not k.endswith('.attn.bias')] # ignore these

        # The OpenAI implementation uses a conv1d rather than a vanilla linear layer, so they need to be transposed
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"missing keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the conv1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy as-is
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model