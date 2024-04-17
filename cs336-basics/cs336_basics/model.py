#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import softmax

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x


class BasicsTransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: Optional[float], default is None.
            If given, apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie the weights, since the paper mentions that "we share the same weight
        # matrix between the two embedding layers and the pre-softmax linear transformation"
        self.lm_head.weight = self.token_embeddings.weight
        self.residual_pdrop = residual_pdrop
        # report number of parameters
        logger.info(
            "number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: LongTensor of shape `(batch_size, sequence_length)`.
                Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        _, sequence_length = x.size()
        # (batch size, sequence_length, d_model)
        # NOTE: paper mentions "In the embedding layers, we multiply those
        # weights by sqrt(d_model)", but we aren't doing that here.
        embedded_tokens = self.token_embeddings(x)

        # Shape: (1, sequence_length)
        positions = torch.arange(
            0, sequence_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        # (1, sequence_length, d_model)
        embedded_positions = self.position_embeddings(positions)
        # (batch size, sequence_length, d_model)
        x = embedded_tokens + embedded_positions
        if self.residual_pdrop:
            # (batch size, sequence_length, d_model)
            x = F.dropout(x, self.residual_pdrop)
        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(
                    topk_mask, float("-inf")
                )
            next_token_probabilities = softmax(
                temperature_scaled_next_token_logits, dim=-1
            )
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: Optional[float], default is None.
            If given, apply dropout to the output of each sub-layer, before it is added
            to the sub-layer input and normalized (section 5.4).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_pdrop=attn_pdrop,
        )
        self.ln1 = RMSNorm(d_model)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model)
        self.residual_pdrop = residual_pdrop

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_attn = self.attn(self.ln1(x))
        if self.residual_pdrop is not None:
            x_attn = F.dropout(x_attn, self.residual_pdrop)
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        if self.residual_pdrop is not None:
            x_ffn = F.dropout(x_ffn, self.residual_pdrop)
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = gelu(x)
        x = self.w2(x)
        return x


def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.Tensor] = None,
    pdrop: Optional[float] = None,
):
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    d_k = K.size(-1)
    # Shape: (batch_size, sequence_length, sequence_length)
    attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    # Apply the mask, if we have one.
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    # NOTE: This dropout isn't really mentioned in the paper (besides the start of section 6.3,
    # "We performed only a small number of experiments to select the dropout, both
    # attention and residual"), but it appears in T2T.
    # https://stats.stackexchange.com/questions/509798/attention-dropout-where-was-it-proposed-used-first
    if pdrop is not None:
        attention_weights = F.dropout(attention_weights, p=pdrop)
    # Shape: (batch_size, sequence_length, d_v)
    return attention_weights @ V


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causl multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        attn_pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop

        self.d_k = int(d_model / num_heads)
        self.d_v = self.d_k

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.d_v, bias=False)
        # W_{O} in the Transformer paper.
        self.output_proj = nn.Linear(
            self.num_heads * self.d_v, self.d_model, bias=False
        )

    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to perform multi-headed self-attention on.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        batch_size, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        # Project the keys, queries, and values for grouped-query attention.
        # Q is of shape (batch_size, seq_len, num_heads * key_dim)
        Q = self.q_proj(x)
        # K is of shape (batch_size, seq_len, num_key_value_heads * key_dim)
        K = self.k_proj(x)
        # V is of shape (batch_size, seq_len, num_key_value_heads * key_dim)
        V = self.v_proj(x)

        # Reshape Q, K, V to (batch_size, num_heads, seq_len, d_k).
        # First, we reshape from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_k)
        # Then, we transpose to go from (batch_size, seq_len, num_heads, d_k) to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(
            1, 2
        )
        K = K.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(
            1, 2
        )
        V = V.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(
            1, 2
        )
        # TODO(nfliu): check if register_buffer is faster?
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device), diagonal=1
        ).bool()
        # Shape: (batch_size, num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(
            K=K, Q=Q, V=V, mask=causal_mask, pdrop=self.attn_pdrop
        )
        # Now, we need to "concat" the outputs of the different heads by reshaping to
        # (batch_size, sequence_length, num_heads * d_v).
        # First, we need to undo the earlier transpose to go from (batch_size, num_heads, sequence_length, d_v)
        # to (batch_size, sequence_length, num_heads, d_v)
        # Then, we combine the last two dimensions via reshaping to (batch_size, sequence_length, num_heads * d_v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_v * self.num_heads)
        )
        # Apply the output projection
        output = self.output_proj(attn_output)
        return output


def gelu(x: torch.FloatTensor):
    return x * (0.5) * (1 + torch.erf(x / math.sqrt(2)))
