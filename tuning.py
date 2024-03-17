import os
import re
from optparse import OptionParser

import torch
from tqdm import tqdm

# High bandwidth model downloading
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Force CPU-only training
# os.environ["CUDA_VISIBLE_DEVICES"]=""

from transformers import AutoTokenizer
from datasets import load_dataset, Split
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import pipeline
from peft import LoraConfig

from utils import TrainerArgs
from utils import TRAINER_DEFAULTS, QUERY_DEFAULTS, OBJECTIVES

def format_target_query(query):
    return QUERY_DEFAULTS.target_prefix + query + QUERY_DEFAULTS.target_postfix

def format_supervisor_prompt(query, response, objective_strs):
     return QUERY_DEFAULTS.supervisor_prefix + f'"{query}"' + QUERY_DEFAULTS.supervisor_infix + f'"{response}"' + QUERY_DEFAULTS.supervisor_postfix.format(*objective_strs)

def tokenize(sample):
    sample['input_ids'] = target_tokenizer.encode(format_target_query(sample['query']), max_length=256)#, truncation=True, padding='max_length')#, return_tensors='pt')
    return sample

def collate(data):
    return {key: [d[key] for d in data] for key in data[0]}

def get_reward(reward_text):
    obj_reward = float(re.findall('[0-9]+', reward_text[0]['generated_text'])[0])
    assert 1 <= obj_reward <= 10, 'Invalid objective reward provided.'
    return torch.tensor(obj_reward)

def initialize_option_parser():
    """
    Initializes the option parser
    :return: the option parser
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs',
                      action='store',
                      type='int',
                      default=TRAINER_DEFAULTS.epochs,
                      dest='epochs',
                      help='Number of epochs to tune for.')
    parser.add_option('-b', '--batch-size',
                      action='store',
                      type='int',
                      default=TRAINER_DEFAULTS.batch_size,
                      dest='batch_size',
                      help='The number of samples per tuning batch (steps applied per-batch).')
    parser.add_option('-m', '--mini-batch-size',
                      action='store',
                      type='int',
                      default=TRAINER_DEFAULTS.mini_batch_size,
                      dest='mini_batch_size',
                      help='The number of samples per mini-batch of each tuning batch (used by PPO trainer).')
    parser.add_option('-o', '--step-each-objective',
                      action='store_true',
                      default=False,
                      dest='step_each_objective',
                      help='Indicate whether the bootstrap-tuner should step for each objective.'
                           'False: Sum all objectives and step once.'
                           'True: Step for each objective.')
    parser.add_option('-d', '--out-dir',
                      action='store',
                      type='str',
                      default=TRAINER_DEFAULTS.out_dir,
                      dest='out_dir',
                      help='Path to directory to save output files to.')
    parser.add_option('-t', '--target-model',
                      action='store',
                      type='str',
                      default='ericzzz/falcon-rw-1b-instruct-openorca',  # 'NousResearch/Llama-2-7b-hf'
                      dest='target_model',
                      help='Model to fine tune.')
    parser.add_option('-s', '--bootstrap-model',
                      action='store',
                      type='str',
                      default='ericzzz/falcon-rw-1b-instruct-openorca',  # 'NousResearch/Llama-2-7b-hf'
                      dest='bootstrap_model',
                      help='Model to bootstrap fine-tune with.')

    return parser


# Modified from https://huggingface.co/docs/trl/main/en/ppo_trainer
if __name__ == '__main__':
    # Initialize option parser to manage CLI arguments
    parser = initialize_option_parser()
    options, _ = parser.parse_args()

    trainer_args = TrainerArgs(
        epochs=options.epochs,
        batch_size=options.batch_size,
        mini_batch_size=options.mini_batch_size,
        step_each_objective=options.step_each_objective,
        out_dir=options.out_dir
    )

    target_model_id = options.target_model
    bootstrap_model_id = options.bootstrap_model

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

    target_tokenizer.pad_token_id = target_tokenizer.eos_token_id

    reward_model = pipeline('text-generation', model=bootstrap_model_id, device_map='auto')

    # Load TruthfulQA dataset. VALIDATION is the only available split
    dataset = load_dataset(path='truthful_qa', name='generation', split=Split.VALIDATION).train_test_split(train_size=0.66, shuffle=True, seed=42)
    dataset = dataset.rename_column('question', 'query')
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type='torch')

    config = PPOConfig(
        model_name=target_model_id,
        # accelerator_kwargs=dict(mixed_precision='fp16'),
        batch_size=trainer_args.batch_size,
        mini_batch_size=trainer_args.mini_batch_size,
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

    for epoch in range(trainer_args.epochs):
        for batch in tqdm(trainer.dataloader, desc=f'Epoch {epoch+1}/{trainer_args.epochs}'):
            query_tensors = batch['input_ids']

            #### Get response from SFTModel
            response_tensors = trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
            batch['response'] = target_tokenizer.batch_decode(response_tensors)

            rewards = [torch.tensor(0) for _ in range(len(query_tensors))]  # This does nothing if STEP_EACH_OBJECTIVE is true

            for i, objective in enumerate(OBJECTIVES):
                reward_prompts = [format_supervisor_prompt(q, r, objective) for q, r in zip(batch['query'], batch['response'])]

                #### Compute reward score
                pipe_outputs = reward_model(reward_prompts, return_full_text=False, max_new_tokens=4, pad_token_id=reward_model.tokenizer.eos_token_id)

                if trainer_args.step_each_objective:
                    rewards = [get_reward(output) for output in pipe_outputs]
                else:
                    rewards = [reward + get_reward(output) for reward, output in zip(rewards, pipe_outputs)]

                #### Run PPO step (run for each objective if desired, otherwise only on sum following last objective
                if trainer_args.step_each_objective or i+1 == len(OBJECTIVES):
                    stats = trainer.step(query_tensors, response_tensors, rewards)
                    trainer.log_stats(stats, batch, rewards)

    model_out = f'{trainer_args.out_dir}/models'
    os.makedirs(model_out, exist_ok=True)
    trainer.save_pretrained(f'{model_out}/{target_model_id.split("/")[-1]}_bootstrap-trained')
