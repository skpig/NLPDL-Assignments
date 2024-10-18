from lib2to3.pgen2 import token
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
    "Hi, my dog is cute",
    "Hello, my cat is cute",
    "Hi, my dog is cute. Hi, my cat is cute",
    "Hello, my dog is cute. Hello, my cat is cute. Hello, my dog is cute. Hello, my cat is cute",
]


@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    for timestep in range(50):
        outputs = custom_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)

    return tokenized_batch['input_ids']

@torch.no_grad()
def golden_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    for timestep in range(50):
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)
    
    return tokenized_batch['input_ids']

bsz = 4
for i in range(0, (len(dataset_with_different_lengths) + bsz - 1) // bsz):
    batch = dataset_with_different_lengths[i * bsz: (i + 1) * bsz]
    golden_res = golden_greedy_decoding(batch)
    custom_res = customized_greedy_decoding(batch)

    assert torch.allclose(golden_res, custom_res), "Decoding results are not equal"
