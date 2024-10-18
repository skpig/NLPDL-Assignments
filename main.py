from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from copy import deepcopy

from customized_gpt2 import CustomizedGPT2LMHeadModel

# tokenizer = AutoTokenizer.from_pretrained("/data2/pretrain/Qwen/Qwen2-0.5B/")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
# config = AutoConfig.from_pretrained("openai-community/gpt2", _attn_implementaion='eager')
original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager").to('cuda')
# original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", config=config).to('cuda')
custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").to('cuda')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to('cuda')

dataset_with_different_lengths = [
    "Hello, my dog is cute",
    "Hello, my cat is cute",
    "Hello, my dog is cute. Hello, my cat is cute",
    "Hello, my dog is cute. Hello, my cat is cute. Hello, my dog is cute. Hello, my cat is cute",
    "Hello, my dog is cute. Hello, my cat is cute. Hello, my dog is cute. Hello, my cat is cute. Hello, my dog is cute. Hello, my cat is cute",
]


@torch.no_grad()
def customized_greedy_decoding(tokenized_batch):
    ################################
    # TODO: Fill in your code here
    # Hint: initialize your past key value structure here
    ################################
    past_key_values = None

    for _ in range(20):
        outputs = custom_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs.logits, dim=-1)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens[:, -1:]], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens[:, -1:])], dim=-1)
        return outputs

@torch.no_grad()
def greedy_decoding(tokenized_batch):
    ################################
    # TODO: Fill in your code here
    # Hint: initialize your past key value structure here
    ################################
    past_key_values = None

    for _ in range(20):
        outputs = original_model(**tokenized_batch, past_key_values=past_key_values)
        custom_outputs = custom_model(**tokenized_batch, past_key_values=past_key_values)
        assert torch.allclose(outputs.logits, custom_outputs.logits, atol=1e-3), "Logits are not equal"
        output_tokens = torch.argmax(outputs.logits[:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = output_tokens
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)
        past_key_values = outputs.past_key_values

    # return outputs

bsz = 2
for i in range(0, (len(dataset_with_different_lengths) + bsz - 1) // bsz):
    batch = dataset_with_different_lengths[i * bsz: (i + 1) * bsz]
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    greedy_decoding(tokenized_batch)