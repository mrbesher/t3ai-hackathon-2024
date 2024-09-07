import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split

BASE_SYSTEM_IF_NOT_EXIST = "Sen yardımcı bir asistansın. Dışarıdan bir fonksiyona erişimin yok."
SYSTEM_WITH_FUNCTIONS = "Sen aşağıdaki fonksiyonlara erişimi olan yardımcı bir asistansın. Kullanıcı sorusuna yardımcı olabilmek için bir veya daha fazla fonksiyon çağırabilirsin. Fonksiyon parametreleri ile ilgili varsayımlarda bulunma. Her fonksiyon çağrısı fonksiyon ismi ve parametreleri ile olmalıdır. İşte, kullanabileceğin fonksiyonlar:"


def format_fnc_convos(convo):
    if isinstance(convo, list):
        system_message = {"role": "system", "content": BASE_SYSTEM_IF_NOT_EXIST}
        convo.insert(0, system_message)
        return convo
    tools = convo["tools"]
    chat = convo["translated_chat"]
    if tools:
        system_message = {"role": "system", "content": SYSTEM_WITH_FUNCTIONS, "tools": tools}
    else:
        system_message = {"role": "system", "content": BASE_SYSTEM_IF_NOT_EXIST}
    chat.insert(0, system_message)
    return chat


DATA_WOUT_FUNCTIONS_PATH = ""
DATA_WITH_FUNCTIONS_PATH = ""

train_data = []
with open(DATA_WOUT_FUNCTIONS_PATH, "r") as f1:
    for line in f1:
        train_data.append(format_fnc_convos(json.loads(line)))

train_data_w_funcs = []
with open(DATA_WITH_FUNCTIONS_PATH, "r") as f1:
    for line in f1:
        train_data_w_funcs.append(format_fnc_convos(json.loads(line)))

tools = [{'name': 'get_exchange_rate',
          'description': 'Get the exchange rate between two currencies',
          'parameters': {'type': 'object',
                         'properties': {'base_currency': {'type': 'string',
                                                          'description': 'The currency to convert from'},
                                        'target_currency': {'type': 'string',
                                                            'description': 'The currency to convert to'}},
                         'required': ['base_currency', 'target_currency']}}, {"a": "b"}]
tools = "\n\n".join([json.dumps(tool) for tool in tools])

from transformers import AutoTokenizer
from huggingface_hub import HfFolder

tokenizer_path = ""
model_path = ""

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

for convo in train_data_w_funcs:
    for turn in convo:
        if turn["role"] == "function_call":
            turn["role"] = "tool_call"
        elif turn["role"] == "function_response":
            turn["role"] = "tool_response"

from transformers import AutoModelForCausalLM

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup, BitsAndBytesConfig
import random
import numpy as np
import time
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
import torch
from tqdm import tqdm
import argparse

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained("t3ai-org/pt-model", device_map="auto")

print('enabling gradient checkpointing')
model.gradient_checkpointing_enable()
print('preparing model for kbit training')
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

funcs_df = pd.DataFrame({"text": train_data_w_funcs})
gen_df = pd.DataFrame({"text": train_data})
gen_train, gen_test = train_test_split(gen_df, test_size=0.70)
funcs_train, funcs_test = train_test_split(funcs_df, test_size=0.90)
train = pd.concat([funcs_train, gen_train])
test = pd.concat([funcs_test, gen_test])

train.to_csv("train_run_1.csv", index=False)
test.to_csv("test_run_1.csv", index=False)

funcs_sample = random.sample(train_data_w_funcs, 500)
gen_sample = random.sample(train_data, 500)

gen_sample.extend(funcs_sample)

_train_data = [tokenizer.apply_chat_template(sample, tokenize=False) for sample in train.text]

from functools import partial


def create_prompt_formats(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    # DEFAULT_SYSTEM_PROMPT = "Sen yardımcı bir asistansın ve sana verilen talimatlar doğrultusunda en iyi cevabı üretmeye çalışacaksın."

    # TEMPLATE = (
    #     "[INST] {system_prompt}\n\n"
    #     "{instruction}[/INST]\n\n"
    #     "Output: {output}"
    # )

    sample["text"] = sample["full_text"]

    return sample


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (str): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)
    print('dataset: ', dataset)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
    )

    # Filter out samples that have "input_ids" exceeding "max_length"

    print('dataset2: ', dataset)
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


full_df = pd.DataFrame({"full_text": _train_data})

full_df.to_csv("train_t3_llm.csv", index=False)

from datasets import load_dataset, Dataset

dataset = Dataset.from_pandas(full_df)

tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token = False
tokenizer.add_bos_token, tokenizer.add_eos_token

max_length = 896
seed = 1337

preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)

print(preprocessed_dataset)

print(preprocessed_dataset[0])

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model_path,
    tokenizer=tokenizer,
    train_dataset=preprocessed_dataset,
    dataset_text_field="text",
    max_seq_length=896,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        learning_rate=2e-4,
        logging_steps=5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
