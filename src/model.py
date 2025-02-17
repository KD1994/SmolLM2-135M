import math
import logging
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Root Mean Square Layer Normalization
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)

        t = torch.arange(max_seq_len, dtype=self.freqs.dtype)
        freqs = torch.outer(t, self.freqs)

        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)


    def rotate_half(self, x):
        rot_dim = x.shape[-1]
        x1 = x[..., :rot_dim // 2]
        x2 = x[..., rot_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, t, x):
        rot_dim = self.freqs.shape[-1]
        cos = self.cos[t, :rot_dim]
        sin = self.sin[t, :rot_dim]

        rotated_x = (x[..., :rot_dim] * cos) + (self.rotate_half(x[..., :rot_dim]) * sin)
        if x.shape[-1] > rot_dim:
            rotated_x = torch.cat((rotated_x, x[..., rot_dim:]), dim=-1)
        return rotated_x
    
    def forward(self, x, seq_dim=-2):
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device)
        return self.apply_rotary_emb(t, x)


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.kv_head_dim = args.dim // args.n_kv_heads

        assert self.head_dim * args.n_heads == args.dim, "args.dim must be divisible by args.n_heads"
        assert self.kv_head_dim * args.n_kv_heads == args.dim, "args.dim must be divisible by args.n_kv_heads"

        self.query_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.key_proj = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim)
        
        self.out_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

        # # Caching storage (keys and values)
        cached_keys = None
        cached_values = None
        self.register_buffer('cached_keys', cached_keys)
        self.register_buffer('cached_values', cached_values)

    def forward(self, x, mask=None, use_cache=False):
        # # batch_size = x.size(0)
        batch_size, seq_len, C = x.size()
        
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape for attention computation
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.kv_heads, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        value = value.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]

        query = self.rope(query)
        key = self.rope(key)

        # # If kv_heads are less than num_heads, repeat them
        # if self.kv_heads < self.num_heads:
        #     key = key.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        #     value = value.repeat_interleave(self.num_heads // self.kv_heads, dim=1)
        
        # # Compute attention
        # attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     attn_weights = attn_weights + mask
        # attn_weights = F.softmax(attn_weights, dim=-1)
        
        # # Compute output
        # output = torch.matmul(attn_weights, value)

        # Flash-attn
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=self.dropout.p, enable_gqa=True)
        
        # Update cache only if using cache
        if use_cache:
            self.cached_keys = key
            self.cached_values = value
        else:
            # Reset cached values during training (to prevent unwanted accumulation)
            self.cached_keys = None
            self.cached_values = None

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch, seq_len, num_heads * head_dim]
        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, args):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.    # 2304
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (nn.Linear): Linear transformation for the first layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third layer.

        """
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.intermediate_dim, bias=False)
        self.w2 = nn.Linear(args.intermediate_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_cache: bool
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
            use_cache (bool): whether to use kv_cache

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), mask=mask, use_cache=use_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args):
        """
        Initialize a Transformer model.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            args (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (nn.Embedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (nn.Linear): Linear layer for final output.

        """
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.output = nn.Linear(
        #     args.dim, args.vocab_size, bias=False
        # )

        # # weight sharing
        # self.output.weight = self.tok_embeddings.weight

        # weight initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        std = self.args.init_scale
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            # if module.bias is not None:
            #     module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = False):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
            use_cache (bool): whether to use kv_cache

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if mask is None:
            mask = torch.triu(torch.ones((seqlen, seqlen), 
                                         dtype=torch.bool, 
                                         device=tokens.device), 
                              diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask * -1e4
        
        for layer in self.layers:
            h = layer(h, mask, use_cache)
        h = self.norm(h)
        # output = self.output(h).float()
        output = F.linear(h, self.tok_embeddings.weight)
        return output

    def generate(self, 
        input_ids, 
        max_length, 
        min_length=None,
        num_return_sequences=1, 
        pad_token_id=None,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    ):
        self.eval()
        # batch_size = input_ids.shape[0]
        min_length = min_length if min_length is not None else input_ids.shape[1]
                   
        with torch.no_grad():
            for ret_seq in range(num_return_sequences):
                logger.info(f"Sequence #{ret_seq + 1}:")
                for _ in range(max_length - input_ids.shape[1]):
                    outputs = self(input_ids, use_cache=True)
                    next_token_logits = outputs[:, -1, :]
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    if do_sample:
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                    
                    input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                    
                    # Stop if all sequences have hit the pad token
                    if pad_token_id is not None and (next_tokens == pad_token_id).all():
                        break
                    
                    # Stop if we've reached min_length
                    if input_ids.shape[1] < min_length:
                        continue
                    
        return input_ids 
