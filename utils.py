from dataclasses import dataclass
import re

import torch

@dataclass
class TrainerArgs:
    epochs: int = 5
    batch_size: int = 32
    mini_batch_size: int = 8
    step_each_objective: bool = False
    out_dir: str = './out'
    cache_dir: str = None

@dataclass
class QueryArgs:
    target_prefix: str = 'PROMPT: '
    target_postfix: str = '\nRESPONSE: '

    supervisor_prefix: str = 'CONTEXT: A generative AI assistant was asked to respond to the following query: '
    supervisor_infix: str = ' The AI assistant provided the following response: '
    supervisor_postfix: str = '\nTASK: Rate the AI assistant\'s response above in terms of its {} on a scale from 1 ({}) to 10 ({})' + \
                              '\nRATING: '

TRAINER_DEFAULTS = TrainerArgs()
QUERY_DEFAULTS = QueryArgs()

OBJECTIVES = [
    ('accuracy', 'highly inaccurate', 'highly accurate'),
    ('clarity', 'very unclear', 'very clear'),
    ('conciseness', 'very unconcise', 'very concise'),
    ('safety', 'very harmful', 'very safe'),
    ('relevance', 'completely irrelevant', 'fully relevant')
]


def format_target_query(query):
    return QUERY_DEFAULTS.target_prefix + query + QUERY_DEFAULTS.target_postfix


def format_supervisor_prompt(query, response, objective_strs):
    return QUERY_DEFAULTS.supervisor_prefix + f'"{query}"' + QUERY_DEFAULTS.supervisor_infix + f'"{response}"' + QUERY_DEFAULTS.supervisor_postfix.format(
        *objective_strs)


def generate_tokenize_fn(tokenizer, max_length=256):
    def tokenize(sample):
        sample['input_ids'] = tokenizer.encode(format_target_query(sample['query']), max_length=max_length)  # , truncation=True, padding='max_length')#, return_tensors='pt')
        return sample

    return tokenize


def collate(data):
    return {key: [d[key] for d in data] for key in data[0]}


def get_reward(reward_text, as_tensor=False):
    obj_reward = float(re.findall('[0-9]+', reward_text[0]['generated_text'])[0])
    if obj_reward == 0:  # failsafe in case the bootstrap model thinks 0 is an acceptable rating
        obj_reward = 1
    assert 1 <= obj_reward <= 10, f'Invalid objective reward provided: {obj_reward}, pulled from "{reward_text[0]["generated_text"]}"'
    return torch.tensor(obj_reward) if as_tensor else obj_reward
