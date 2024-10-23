from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2AttentionWithKVCache(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values = None,
        **kwargs,
    ):

        # Prepare query, key, value matrix
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # each of them has shape (batch_size, seq_len, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]

        # Concat previous key & value
        if past_key_values:
            past_key, past_value = past_key_values
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, (key, value)


class CustomizedGPT2BlockWithKVCache(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2AttentionWithKVCache(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values = None,
        **kwargs,
    ):
        residual = hidden_states

        # self-attention (class `CustomizedGPT2AttentionWithFasterCache`)
        hidden_states = self.ln_1(hidden_states)
        attn_output, past_key_values = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        # residual connection
        hidden_states = attn_output + residual


        residual = hidden_states

        # feed-forward
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states


        return hidden_states, past_key_values


class CustomizedGPT2ModelWithKVCache(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2BlockWithKVCache(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            past_len = past_key_values[0][0].shape[-2]
            position_ids = position_ids[:,past_len:]
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds


        # Prepare Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)


        cur_past_key_values = []
        # Iterate over all GPT2 layer, i.e. `block`
        for i, block in enumerate(self.h):
            outputs, cur_layer_past_key_values = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values[i] if past_key_values else None,
            )

            hidden_states = outputs
            cur_past_key_values.append(cur_layer_past_key_values)


        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states, cur_past_key_values


class CustomizedGPT2LMHeadModelWithKVCache(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2ModelWithKVCache(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values = None,
    ):
        hidden_states, past_key_values = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )

        # Prepare logits from last hidden state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'past_key_values': past_key_values
        }