import json
import math
import os

from tqdm import tqdm

import copy
import glob
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.distributed as dist
import transformers
import wandb
from datasets import disable_caching
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed
from transformers.integrations.deepspeed import deepspeed_config

from train_utils import LogCallback, PyTorchProfilerCallback, print_rank0
from trainer import MinimalTrainer
from eval_utils import download

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="kaizen9/phi-1_5_HQ_6000_200k_FP")
    flash_attention: Optional[bool] = field(default=True)
    config_dtype: Optional[str] = field(default="bfloat16")

    # z-loss
    z_loss: float = field(default=0.0, metadata={"help": "z_loss"})

    # gradient checkpointing
    gradient_checkpointing_step: int = field(
        default=7, metadata={"help": "gradient_checkpointing_step"})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    log_dir: str = field(default=None)
    profile: bool = field(default=False)
    # NOTE: insert the 8bit adamw etc here. I think we want to just go linear and copy the og paper
    # also need to tune lr a little 
    learning_rate: float = 6e-5
    max_grad_norm: float = 1.0
    #warmup_ratio: float = 0.05 # or however much is like 500 steps
    warmup_steps: int = 200
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 1.0
    max_steps: int =6000
    optim: str = "adamw_torch_fused"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # optimize performance and memory
    per_device_eval_batch_size: int = 16  # TODO: auto-find?
    gradient_checkpointing: bool = True
    bf16: bool = True
    #resume_from_checkpoint : str= "./phi/checkpoint-18"
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit = 10
    eval_strategy: str = "steps"
    eval_steps: int = 500

    update_trained_steps_and_epochs: bool = field(  # whether to start a new curriculum phase
        default=False,
        metadata={
            "help":
            "Update the trainer state with the trained steps and epochs."  # 每一轮开始时更新读取的step和epoch，否则从模型的config.json中读取
        })
    num_steps_trained_before_this_epoch: int = field(
        default=0,
        metadata={"help": "多少步在这个epoch之前训过"})  # /home/u20140041/pretrain-mini/.venv/lib/python3.12/site-packages/transformers/trainer.py:2168
    num_epochs_trained_before_this_epoch: int = field(
        default=0,
        metadata={"help": "多少个epoch在这个epoch之前训过"})
    log_dir: str = field(default=None)
    profile: bool = field(default=False)
    remove_unused_columns: Optional[bool] = False

pass




class PretrainDataset(Dataset):

    # To use doc attention mask, add "position_ids" column to the `train_dataset`

    def __init__(self, train_dataset, eos_token_id = None, skip=0):
        super(PretrainDataset, self).__init__()
        self.sources = train_dataset.shuffle(seed=42)
        self.banned_idx = set()  # <-- Add banned indices here (not used)
        self.available_idx = []
        self.sorted_banned = sorted(self.banned_idx)
        self.skip = skip
        self.size = len(self.sources) - len(self.banned_idx) + skip
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        idx -= self.skip
        try:
            f = self.sorted_banned.index(idx)
        except ValueError:
            f = None
        if f is not None:
            idx = self.available_idx[f]

        ipt_ids = self.sources[idx]["input_ids"]
        # the logic here doesn't seem to make sense, but internally transformers handles it
        results = dict(input_ids=ipt_ids,
                       labels=copy.deepcopy(ipt_ids),
                       idx=idx)
        # if "position_ids" in self.sources[idx]:  # for doc attention mask
        #     results["position_ids"] = self.sources[idx]["position_ids"]
        # if "subset" in self.sources[idx]:  # record the subset
        #     results["subset"] = self.sources[idx]["subset"]

        return results


@dataclass
class DataCollatorForPretrainDataset:
    data_args: DataArguments
   # tokenizer: transformers.PreTrainedTokenizer
    # pretty simple class
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        r = dict(
            input_ids=torch.tensor([d["input_ids"] for d in instances]),
            labels=torch.tensor([copy.deepcopy(d["input_ids"]) for d in instances]),
            # unclear as to exactly why idx doesn't get collated properly
        )
        # if "position_ids" in instances[0]:  # for doc attention mask
        #     r["position_ids"] = torch.tensor(
        #         [d["position_ids"] for d in instances])
        return r


def prepare_data(tokenizer,
                                 data_args,
                                training_args,
                                model_args,
                                skip=0):

    # staggered start
    sleep_time = RANK
    time.sleep(sleep_time / 2)

    # load the dataset
    # since we're following LLM 360 for data, we need to change this
    #data_files = ['train/train_0.jsonl', 'train/train_1.jsonl']#, 'train/train_3.jsonl', 'train/train_4.jsonl']
    data_files = ['train/train_13.jsonl', 'train/train_11.jsonl', 'train/train_12.jsonl']

    train_dataset = datasets.load_dataset("semran1/packed_20B", split="train", data_files=data_files)
    train_dataset = train_dataset.rename_column("token_ids", "input_ids")
    print(f"train dataset size: {len(train_dataset)}")
    train_dataset = PretrainDataset(train_dataset=train_dataset,
                                     eos_token_id=tokenizer.eos_token_id,
                                     skip=skip)
    data_collator = DataCollatorForPretrainDataset(data_args=data_args)
    print("starting eval download")
    val_dataset = datasets.load_dataset("semran1/packed_20B", split="train", data_dir = "valid")
    val_dataset = val_dataset.rename_column("token_ids", "input_ids")
    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


def get_model_tokenizer(model_args, data_args, training_args):
    from transformers import AutoTokenizer

    from configuration_phi import PhiConfig
    from modeling_phi import QPhiForCausalLM

    # load model and tokenizer
    # config = QPhiConfig.from_pretrained(model_name_or_path)
    model_name_or_path=model_args.model_name_or_path
    model = QPhiForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2"
        if model_args.flash_attention else None,
        torch_dtype=getattr(torch, model_args.config_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    if RANK == 0:
        print("params", sum(p.numel() for p in model.parameters()))

        print(model)
        print(model.model.embed_tokens.weight.data.norm())
        print(tokenizer)
        print(model.config)

    assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"
    return model, tokenizer


def train():
    if RANK == 0:
        wandb.login(key="Bd8e3eb1919976284f0e2615608b1d7af8bdf98b")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # automatically find the latest checkpoint to resume from. compatible with torchrun --max_restarts
    if not re.match(r".*/checkpoint-\d+", model_args.model_name_or_path):
        checkpoints = [
            c
            for c in glob.glob(f"{model_args.model_name_or_path}/checkpoint-*")
            if os.path.basename(c).split("-")[1].isdigit()
        ]
        print("HERE ARE THE CHECKPOINTS", checkpoints)
        if len(checkpoints) > 0:
            model_args.model_name_or_path = max(
                checkpoints, key=lambda x: int(os.path.basename(x).split("-")[1]))
            training_args.resume_from_checkpoint = model_args.model_name_or_path

    # Calculate where to resume training, i.e. new phase or same phase
    # if training_args.num_steps_trained_before_this_epoch != 0 or training_args.num_epochs_trained_before_this_epoch != 0:
    #     raise ValueError(
    #         "num_steps_trained_before_this_epoch and num_epochs_trained_before_this_epoch 不应该从命令行传入")
    # elif training_args.update_trained_steps_and_epochs:  # new curriculum phase
    #     with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")) as f:
    #         state = json.load(f)
    #     training_args.num_steps_trained_before_this_epoch  = state["global_step"]
    #     training_args.num_epochs_trained_before_this_epoch = math.ceil(state["epoch"])
    #     del state
    # # else:  # continue training in the same curriculum phase
    # #     with open(os.path.join(training_args.resume_from_checkpoint, "config.json")) as f:
    # #         config = json.load(f)
    # #     training_args.num_steps_trained_before_this_epoch  = config["num_steps_trained_before_this_epoch"]
    # #     training_args.num_epochs_trained_before_this_epoch = config["num_epochs_trained_before_this_epoch"]
    # #     del config

    print("=="*50)
    print(training_args)
    print("=="*50)
    print(training_args.num_epochs_trained_before_this_epoch)
    print(training_args.num_train_epochs)

    assert int(training_args.num_epochs_trained_before_this_epoch) == int(training_args.num_train_epochs - 1), "only allow 1 new epochs"

    # Log the config
    from pprint import pprint
    if RANK == 0:
        print(f"Resuming from {model_args.model_name_or_path}")
        pprint(model_args.__dict__)
        pprint(data_args.__dict__)
        pprint(training_args.__dict__)

        config = {"rank": RANK}
        config.update(model_args.__dict__)
        config.update(data_args.__dict__)
        for key, value in training_args.__dict__.items():
            try:
                json.dumps(value)
            except Exception:
                print(
                    f"Key '{key}' contains non-serializable value: {value} (type: {type(value)})"
                )
                continue
        config.update({key: value})

        wandb_path = "log-wandb/" # training_args.log_dir.replace("log/", "log-wandb/", 1)
        #if RANK == 0:
        print(f"wandb_path: {wandb_path}")
        os.makedirs(wandb_path, exist_ok=True)
         #if RANK == 0:
        wandb.init(project="qphi",
               resume="allow",
               group=training_args.run_name,
               name=training_args.run_name,
               config=config,
               dir=wandb_path)
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }  # OR gradient_checkpointing_kwargs={'use_reentrant':True}, please refer to https://github.com/huggingface/transformers/issues/26969

    # Get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_args, data_args,
                                           training_args)

    if training_args.update_trained_steps_and_epochs:
        model.config.num_steps_trained_before_this_epoch  = training_args.num_steps_trained_before_this_epoch
        model.config.num_epochs_trained_before_this_epoch = training_args.num_epochs_trained_before_this_epoch
    model.config.gradient_checkpointing_step = model_args.gradient_checkpointing_step

    set_seed(training_args.seed)
    training_args.config = model.config
    dscfg = deepspeed_config()
    if RANK == 0:
        print(dscfg)

    # Modify `trainer_state.json`. This is necessary because HF Trainer, when there's a conflict between CLI arguments and parameters in the JSON file, prioritizes the JSON parameters. Therefore, we need to manually overwrite the JSON parameters with the CLI arguments.
    if training_args.resume_from_checkpoint:
        
        with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")) as f:
            state = json.load(f)
        print("loaded checkpoint:", training_args.resume_from_checkpoint)
        torch.distributed.barrier()

        if state["train_batch_size"] != training_args.per_device_train_batch_size:
            if RANK == 0:
                print("Warning: train_batch_size is different from the checkpoint")
                state["train_batch_size"] = training_args.per_device_train_batch_size
                if not os.path.exists(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json.bak")):
                    shutil.copyfile(
                        os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"),
                        os.path.join(training_args.resume_from_checkpoint, "trainer_state.json.bak"),
                    )
                with open(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"), "w") as fn:
                    fn.write(json.dumps(state, indent=2))
            torch.distributed.barrier()

    # Prepare the data
    data_module = prepare_data(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
        skip=0,
    )

    # Update max_steps based on the length of the dataset
    num_update_steps_per_epoch = math.ceil((len(data_module["train_dataset"])/(WORLD_SIZE * training_args.per_device_train_batch_size)))// training_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    print("num_update_steps_per_epoch: ")
    print(training_args.num_steps_trained_before_this_epoch)
    print(num_update_steps_per_epoch)
    print(len(data_module["train_dataset"]))
    print(WORLD_SIZE)
    print(training_args.per_device_train_batch_size)
    print(len(data_module["train_dataset"]) //(WORLD_SIZE * training_args.per_device_train_batch_size))
    print(training_args.gradient_accumulation_steps)
    training_args.max_steps = training_args.num_steps_trained_before_this_epoch + num_update_steps_per_epoch

    model.tokenizer = tokenizer

    if training_args.profile:
        callbacks = [PyTorchProfilerCallback, LogCallback]
    else:
        callbacks = [LogCallback]

    model.LOCAL_RANK = LOCAL_RANK
    model.log_dir = training_args.log_dir
    model.rank = RANK

    trainer = MinimalTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
