import os
import re
from functools import partial
from itertools import repeat

import torch
from tqdm import tqdm

# High bandwidth model downloading
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Force CPU-only training
# os.environ["CUDA_VISIBLE_DEVICES"]=""


from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Split
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import pipeline

from peft import LoraConfig

# target_model_id = 'NousResearch/Llama-2-7b-hf'
# supervisor_model_id = 'NousResearch/Llama-2-7b-hf'

target_model_id = 'ericzzz/falcon-rw-1b-instruct-openorca'
supervisor_model_id = 'ericzzz/falcon-rw-1b-instruct-openorca'

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    # target_modules=['q_proj', 'k_proj', 'v_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type='CAUSAL_LM'
)

# Download and load models
target_model = AutoModelForCausalLMWithValueHead.from_pretrained(target_model_id, peft_config=lora_config)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

# supervisor_model = AutoModelForCausalLMWithValueHead.from_pretrained(supervisor_model_id, peft_config=lora_config)
# supervisor_tokenizer = AutoTokenizer.from_pretrained(supervisor_model_id)

target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
# supervisor_tokenizer.pad_token_id = supervisor_tokenizer.eos_token_id

reward_model = pipeline('text-generation', model=supervisor_model_id, device_map='auto')


# TARGET_QUERY_PREFIX = '<|system|>' + \
#                       '\nPROMPT: '
# TARGET_QUERY_POSTFIX = '\n<|assistant|>' + \
#                        '\nRESPONSE: '
#
# SUPERVISOR_QUERY_PREFIX = '<|system|>' + \
#                           '\nCONTEXT: A generative AI assistant was asked to respond to the following query:\n"'
#
# SUPERVISOR_QUERY_INFIX = '"\nThe model provided the following response:\n"'
# SUPERVISOR_QUERY_POSTFIX = '"\nTASK: Rate the model\'s response above in terms of its accuracy on a scale from 1 (highly inaccurate) to 10 (highly accurate):' + \
#                            '\n<|assistant|>\nRATING: '
# SUPERVISOR_QUERY_POSTFIX = '\nTask: Rate the previous answer in terms of its accuracy on a scale from 1 (highly inaccurate) to 10 (highly accurate).\nAnswer [in range 1-10]: '

TARGET_QUERY_PREFIX = 'PROMPT: '
TARGET_QUERY_POSTFIX = '\nRESPONSE: '

SUPERVISOR_QUERY_PREFIX = 'CONTEXT: A generative AI assistant was asked to respond to the following query: '
SUPERVISOR_QUERY_INFIX = ' The model provided the following response: '
SUPERVISOR_QUERY_POSTFIX = '\nTASK: Rate the model\'s response above in terms of its {} on a scale from 1 ({}) to 10 ({})' + \
                           '\nRATING: '

OBJECTIVES = [
    ('accuracy', 'highly inaccurate', 'highly accurate'),
    ('clarity', 'very unclear', 'very clear'),
    ('conciseness', 'very unconcise', 'very concise'),
    ('safety', 'very unsafe', 'very safe')
]

def format_target_query(query):
    return TARGET_QUERY_PREFIX + query + TARGET_QUERY_POSTFIX

def format_supervisor_prompt(query, response, objective_strs):
     return SUPERVISOR_QUERY_PREFIX + '"' + query + '"' + SUPERVISOR_QUERY_INFIX + '"' + response + '"' + SUPERVISOR_QUERY_POSTFIX.format(*objective_strs)

def tokenize(sample):
    sample['input_ids'] = target_tokenizer.encode(format_target_query(sample['query']), max_length=256)#, truncation=True, padding='max_length')#, return_tensors='pt')
    return sample

def collate(data):
    return {key: [d[key] for d in data] for key in data[0]}

# Load TruthfulQA dataset. VALIDATION is the only available split
dataset = load_dataset(path='truthful_qa', name='generation', split=Split.VALIDATION).train_test_split(train_size=0.66, shuffle=True, seed=42)
dataset = dataset.rename_column('question', 'query')
dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type='torch')



config = PPOConfig(
    model_name=target_model_id,
    # accelerator_kwargs=dict(mixed_precision='fp16'),
    batch_size=32,
    mini_batch_size=8,
    gradient_accumulation_steps=1
)

trainer = PPOTrainer(
    config=config,
    model=target_model,
    # ref_model=supervisor_model,
    tokenizer=target_tokenizer,
    dataset=dataset[Split.TRAIN],
    data_collator=collate
)

generation_kwargs = dict(
    min_length=-1,
    top_k=0,
    top_p=1.0,
    do_sample=True,
    pad_token_id=target_tokenizer.eos_token_id,
    max_new_tokens=32
)

for batch in tqdm(trainer.dataloader):
    query_tensors = batch['input_ids']

    #### Get response from SFTModel
    response_tensors = trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch['response'] = target_tokenizer.batch_decode(response_tensors)

    rewards = [torch.tensor(0) for _ in range(len(query_tensors))]

    #### Compute reward score
    for objective in OBJECTIVES:
        texts = [format_supervisor_prompt(q, r, objective) for q, r in zip(batch['query'], batch['response'])]
        pipe_outputs = reward_model(texts, return_full_text=False, max_new_tokens=4, pad_token_id=reward_model.tokenizer.eos_token_id)
        rewards = [rewards[i] + 11.-float(re.findall('[1-9]|10', output[0]['generated_text'])[0]) for i, output in enumerate(pipe_outputs)]


    #### Run PPO step
    stats = trainer.step(query_tensors, response_tensors, rewards)
    trainer.log_stats(stats, batch, rewards)

