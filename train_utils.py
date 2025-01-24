import json
import math
import os
from typing import Dict, Union

import datasets
import torch
import transformers
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def print_rank0(*arg):
    if RANK == 0:
        print(*arg)


class LogCallback(TrainerCallback):

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if int(os.getenv("RANK", "0")) == 0:

            logs["train/global_step"] = state.global_step
            logs["train/epoch"] = state.epoch
            logs['train/total_flos'] = state.total_flos
            wandb.config.update({'global_step': state.global_step},
                            allow_val_change=True)


class PyTorchProfilerCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        # only one epoch will be trained
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=20, warmup=0, active=8),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.log_dir),
            record_shapes=True,
            profile_memory=True,
            #with_stack=True,
            with_flops=True,
            #with_modules=True
        )

    def on_step_begin(self, args, state, control, logs=None, **kwargs):
        self.prof.step()

    def on_train_end(self, args, state, control, logs=None, **kwargs):
        self.prof.stop()



def get_wsd_scheduler(optimizer,
                      num_warmup_steps,
                      num_training_steps,
                      last_epoch=-1,
                      stable_ratio=1.0,
                      start_lambda=0,
                      end_lambda=1,
                      start_global_step=None,
                      end_global_step=None,
                      wsd_style="cos"):
    # Note: Due to subsequent modifications to the training code, this function may require re-adaptation.

    if wsd_style == "cos":
        def lr_lambda(current_step):
            if start_global_step is not None and end_global_step is not None and start_global_step <= current_step <= end_global_step:
                return (1 - math.cos(
                    math.pi * float(current_step - start_global_step) /
                    float(max(1, end_global_step - start_global_step)) / 2)) * (
                        end_lambda - start_lambda) + start_lambda
            if current_step < num_warmup_steps:
                return (float(current_step) / float(max(1, num_warmup_steps))) * (
                    end_lambda - start_lambda) + start_lambda
            num_stable_steps = stable_ratio * num_training_steps
            if stable_ratio == 1.0 or current_step <= num_stable_steps:
                return 1.0
            return max(
                0.1,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_stable_steps)),
            )

    elif wsd_style == "linear":
        def lr_lambda(current_step):
            if start_global_step is not None and end_global_step is not None and start_global_step <= current_step <= end_global_step:
                return (float(current_step - start_global_step) /
                        float(max(1, end_global_step - start_global_step))) * (
                            end_lambda - start_lambda) + start_lambda
            if current_step < num_warmup_steps:
                return (float(current_step) / float(max(1, num_warmup_steps))) * (
                    end_lambda - start_lambda) + start_lambda
            num_stable_steps = stable_ratio * num_training_steps
            if stable_ratio == 1.0 or current_step <= num_stable_steps:
                return 1.0
            return max(
                0.1,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_stable_steps)),
            )
    elif wsd_style == "cos2":

        def lr_lambda(current_step):
            if start_global_step is not None and end_global_step is not None and start_global_step <= current_step <= end_global_step:
                return (1 - math.cos(
                    math.pi * float(current_step - start_global_step) /
                    float(max(1, end_global_step - start_global_step)))) * (
                        end_lambda - start_lambda) / 2 + start_lambda
            if current_step < num_warmup_steps:
                return (float(current_step) / float(max(1, num_warmup_steps))
                        ) * (end_lambda - start_lambda) + start_lambda
            num_stable_steps = stable_ratio * num_training_steps
            if stable_ratio == 1.0 or current_step <= num_stable_steps:
                return 1.0
            return max(
                0.1,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_stable_steps)),
            )
    elif wsd_style == "1sqrt":

        def lr_lambda(current_step):
            if current_step > 262000:  # small hack for remaining steps
                current_step = 262000
            if start_global_step is not None and end_global_step is not None and start_global_step <= current_step <= end_global_step:
                return (1 - math.sqrt(
                    (current_step - start_global_step) /
                    (end_global_step - start_global_step))) * (
                        start_lambda - end_lambda) + end_lambda
            if current_step < num_warmup_steps:
                return (float(current_step) / float(max(1, num_warmup_steps))
                        ) * (end_lambda - start_lambda) + start_lambda
            num_stable_steps = stable_ratio * num_training_steps
            if stable_ratio == 1.0 or current_step <= num_stable_steps:
                return 1.0
            return max(
                0.1,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_stable_steps)),
            )
    else:
        raise ValueError(f"Unknown wsd_style: {wsd_style}")

    return LambdaLR(optimizer, lr_lambda, last_epoch)
