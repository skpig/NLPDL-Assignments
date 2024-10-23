import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from copy import deepcopy

from customized_gpt2_answer import CustomizedGPT2LMHeadModelWithKVCache
from customized_gpt2 import CustomizedGPT2LMHeadModel

@torch.no_grad()
def customized_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)

        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time


@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    tokenized_batch['past_key_values'] = None
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model_w_cache(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['past_key_values'] = outputs['past_key_values']
        tokenized_batch['input_ids'] = output_tokens
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)

        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time

@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


@torch.no_grad()
def golden_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
            "past_key_values": outputs['past_key_values']
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 64
    times = [0, 0, 0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")
    custom_model_w_cache = CustomizedGPT2LMHeadModelWithKVCache.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_res, golden_time = golden_greedy_decoding(batch)
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)
        custom_wo_cache_res, custom_wo_cache_time = customized_greedy_decoding_wo_cache(batch)

        times[0] += golden_time
        times[1] += golden_wo_cache_time
        times[2] += custom_time
        times[3] += custom_wo_cache_time

        assert torch.equal(golden_res, custom_res), "Decoding results are not equal"
        assert torch.equal(golden_wo_cache_res, custom_wo_cache_res), "Decoding results are not equal"
        assert torch.equal(golden_res, custom_wo_cache_res), "Decoding results are not equal"

    print("Time taken for golden greedy decoding: ", times[0])
    print("Time taken for golden greedy decoding without cache: ", times[1])
    print("Time taken for customized greedy decoding: ", times[2])
    print("Time taken for customized greedy decoding without cache: ", times[3])
