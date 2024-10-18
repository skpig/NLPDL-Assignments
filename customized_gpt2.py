from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2AttentionWithFasterCache(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ################################
        # TODO: Fill in your code here #
        ################################


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        # Prepare query, key, value based on input_ids (corresponding hidden states)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)



        ###############
        # TODO:(begin)#
        ###############
        # Hint
        # 1. 
        # 2. the `outputs[1:]` returned from `block()` is the return value of your implemented `CustomizedGPT2AttentionWithFasterCache.forward()`
        ###############

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        ##############
        # TODO:(end) #
        ##############






        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        # replace the attention module with the customized one
        attention_class = CustomizedGPT2AttentionWithFasterCache
        self.attn = attention_class(config=config, layer_idx=layer_idx)

class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device




        ###############
        # TODO:(begin)#
        ###############
        # Hint: Prepare the input embeddings based on `input_ids` of current timestep 
        # and `past_key_values` of previous timestep
        ###############

        """ANSWER"""
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        ##############
        # TODO:(end) #
        ##############





        # Prepare Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)




        ###############
        # TODO:(begin)#
        ###############
        # Code analysis
        # 1. the `layer_past` argument passed in block() will be received in your implemented `CustomizedGPT2AttentionWithFasterCache.forward()`
        # 2. the `outputs[1:]` returned from `block()` is the return value of your implemented `CustomizedGPT2AttentionWithFasterCache.forward()`
        ###############
        # Hint
        # 1. You may maintain a `past_key_values` structure (e.g. a List) to store "attention keys" and "attention values" of each layer,
        # and return it for the next

        presents = ()
        ##############
        # TODO:(end) #
        ##############


        # Iterate over all GPT2 layer, i.e. `block`
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,  
                attention_mask=attention_mask,
                use_cache=True,
            )

            hidden_states = outputs[0]
            


            #######
            # TODO:(begin)
            #######


            """ANSWER"""
            # presents = presents + (outputs[1],)

            #######
            # TODO:(end)
            #######


        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
        #         if v is not None
        #     )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()


# class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         config.num_labels = 1
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#         self.multiple_choice_head = SequenceSummary(config)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings(PARALLELIZE_DOCSTRING)
#     def parallelize(self, device_map=None):
#         warnings.warn(
#             "`GPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
#             " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
#             " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
#             " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
#             FutureWarning,
#         )
#         self.device_map = (
#             get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
#             if device_map is None
#             else device_map
#         )
#         assert_device_map(self.device_map, len(self.transformer.h))
#         self.transformer.parallelize(self.device_map)
#         self.lm_head = self.lm_head.to(self.transformer.first_device)
#         self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
#         self.model_parallel = True

#     @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
#     def deparallelize(self):
#         warnings.warn(
#             "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
#             FutureWarning,
#         )
#         self.transformer.deparallelize()
#         self.transformer = self.transformer.to("cpu")
#         self.lm_head = self.lm_head.to("cpu")
#         self.multiple_choice_head = self.multiple_choice_head.to("cpu")
#         self.model_parallel = False
#         torch.cuda.empty_cache()

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, past_key_values=None, **kwargs):
#         token_type_ids = kwargs.get("token_type_ids", None)
#         # Omit tokens covered by past_key_values
#         if past_key_values:
#             past_length = past_key_values[0][0].shape[2]

#             # Some generation methods already pass only the last input ID
#             if input_ids.shape[1] > past_length:
#                 remove_prefix_length = past_length
#             else:
#                 # Default to old behavior: keep only final ID
#                 remove_prefix_length = input_ids.shape[1] - 1

#             input_ids = input_ids[:, remove_prefix_length:]
#             if token_type_ids is not None:
#                 token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

#         attention_mask = kwargs.get("attention_mask", None)
#         position_ids = kwargs.get("position_ids", None)

#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past_key_values:
#                 position_ids = position_ids[:, -input_ids.shape[1] :]
#         else:
#             position_ids = None

#         # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#         if inputs_embeds is not None and past_key_values is None:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids.contiguous()}

#         model_inputs.update(
#             {
#                 "past_key_values": past_key_values,
#                 "use_cache": kwargs.get("use_cache"),
#                 "position_ids": position_ids,
#                 "attention_mask": attention_mask,
#                 "token_type_ids": token_type_ids,
#             }
#         )
#         return model_inputs

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         mc_token_ids: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         mc_labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[Tuple, GPT2DoubleHeadsModelOutput]:
#         r"""
#         mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
#             Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
#             1]`.
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
#             `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
#         mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
#             Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
#             where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

#         Return:

#         Example:

#         ```python
#         >>> import torch
#         >>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel

#         >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#         >>> model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

#         >>> # Add a [CLS] to the vocabulary (we should train it also!)
#         >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
#         >>> # Update the model embeddings with the new vocabulary size
#         >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))

#         >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
#         >>> encoded_choices = [tokenizer.encode(s) for s in choices]
#         >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

#         >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
#         >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

#         >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
#         >>> lm_logits = outputs.logits
#         >>> mc_logits = outputs.mc_logits
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = transformer_outputs[0]

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.transformer.first_device)
#             hidden_states = hidden_states.to(self.lm_head.weight.device)

#         lm_logits = self.lm_head(hidden_states)
#         mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

#         mc_loss = None
#         if mc_labels is not None:
#             loss_fct = CrossEntropyLoss()
#             mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
#         lm_loss = None
#         if labels is not None:
#             labels = labels.to(lm_logits.device)
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#         if not return_dict:
#             output = (lm_logits, mc_logits) + transformer_outputs[1:]
#             if mc_loss is not None:
#                 output = (mc_loss,) + output
#             return ((lm_loss,) + output) if lm_loss is not None else output

#         return GPT2DoubleHeadsModelOutput(
#             loss=lm_loss,
#             mc_loss=mc_loss,
#             logits=lm_logits,
#             mc_logits=mc_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )

#     @staticmethod
#     def _reorder_cache(
#         past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
#     ) -> Tuple[Tuple[torch.Tensor]]:
#         """
#         This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
#         [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
#         beam_idx at every generation step.
#         """
#         return tuple(
#             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
#             for layer_past in past_key_values
#         )


# @add_start_docstrings(
#     """
#     The GPT2 Model transformer with a sequence classification head on top (linear layer).

#     [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-1) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """,
#     GPT2_START_DOCSTRING,
# )
# class GPT2ForSequenceClassification(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.transformer = GPT2Model(config)
#         self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint="microsoft/DialogRPT-updown",
#         output_type=SequenceClassifierOutputWithPast,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size, sequence_length = input_ids.shape[:2]
#         else:
#             batch_size, sequence_length = inputs_embeds.shape[:2]

#         assert (
#             self.config.pad_token_id is not None or batch_size == 1
#         ), "Cannot handle batch sizes > 1 if no padding token is defined."
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#                 sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
#                 sequence_lengths = sequence_lengths % input_ids.shape[-1]
#                 sequence_lengths = sequence_lengths.to(logits.device)
#             else:
#                 sequence_lengths = -1
#                 logger.warning_once(
#                     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#                 )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     GPT2 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
#     Named-Entity-Recognition (NER) tasks.
#     """,
#     GPT2_START_DOCSTRING,
# )
# class GPT2ForTokenClassification(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.transformer = GPT2Model(config)
#         if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
#             classifier_dropout = config.classifier_dropout
#         elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
#             classifier_dropout = config.hidden_dropout
#         else:
#             classifier_dropout = 0.1
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
#     # fmt: off
#     @add_code_sample_docstrings(
#         checkpoint="brad1141/gpt2-finetuned-comp2",
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_loss=0.25,
#         expected_output=[
#             "Lead",
#             "Lead",
#             "Lead",
#             "Position",
#             "Lead",
#             "Lead",
#             "Lead",
#             "Lead",
#             "Lead",
#             "Lead",
#             "Lead",
#             "Lead",
#         ],
#     )
#     # fmt: on
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, TokenClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = transformer_outputs[0]
#         hidden_states = self.dropout(hidden_states)
#         logits = self.classifier(hidden_states)

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + transformer_outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like
#     SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     GPT2_START_DOCSTRING,
# )
# class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.transformer = GPT2Model(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=QuestionAnsweringModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#         real_checkpoint=_CHECKPOINT_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, QuestionAnsweringModelOutput]:
#         r"""
#         start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
#             are not taken into account for computing the loss.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1).to(start_logits.device)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1).to(end_logits.device)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )