import lm_eval
import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaSdpaAttention, LlamaMLP, LlamaDecoderLayer
import argparse
import transformers
from safetensors.torch import load_file
from transformers.modeling_utils import load_sharded_checkpoint

import math
from alg.mergelinear import UniversalPath, Timer
from qlinear import QLinear

class MergeLinear(nn.Linear):
    def __init__(self,
            *kargs,
            **kwargs
        ):
        super(MergeLinear, self).__init__(*kargs, **kwargs)
        """
        This is only for training, and kernel optimization is needed for efficiency.
        """
        self.end_step = 1 #step at which this layer is fully quantized
        self.global_step = 0 #step at which we are currently at
        self.general = UniversalPath(self.out_features, self.in_features)
        self.timer = Timer(self.weight)
        self.norm = LlamaRMSNorm(self.in_features)
        self.loss = None
        self.eval = False


    def forward(self, x):
        """i
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        if self.eval:
            x = self.norm(x)
            y = F.linear(x, self.weight)
            if self.bias is not None:
                y = y + self.bias
            return y
        self.loss = self.get_loss()
        y_dot = self.weight
        x = x.to(y_dot.device)
        #x = self.norm(x)
        # norm scheduling
        w_t = self.general(y_dot, self.get_time())
        # w_t = t*w_t + (1-t)*y_dot
        # introducing weight schduling could be interesting here
        y = F.linear(x, w_t)
        if self.bias is not None:
            y = y + self.bias
        return y
    def get_time(self):
        return self.timer(100)
    def get_loss(self):
        y = self.weight
        c = 1 - 2*y
        x = self.timer.param
        x = x.clamp(0, 1)
        dot = torch.sum(x * c, dim = 1)
        return dot.mean()
    def get_weights(self):
        weight = self.weight
        time = self.timer.param.data.clone()
        time = time.clamp(0, 1)
        time[self.weight < 0] = 1 - time[self.weight < 0]
        # multiply by mask
        time = time * torch.sign(weight)
        scales = self.general.weight.data.clone()
        # scales = scales.values
        print(scales.shape)
        weight = time
        return weight, scales
    
def merge(model1):
    import copy
    # Create a new model by copying model1's structure
    merged_model = copy.deepcopy(model1)
    # time = Timer()
    # Iterate through all modules in the model
    for name, module in merged_model.named_modules():
        # Replace Linear layers with MergeLinear layers
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            # Create new MergeLinear with same dimensions
            merge_layer = MergeLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )

            # Copy weights from model1 to main weights
            path_weight = torch.norm(module.weight.data, dim=1, keepdim=True)
            merge_layer.general.weight.data.copy_(path_weight)
            weight = module.weight.data / path_weight
            merge_layer.weight.data.copy_(weight)
            merge_layer.weight.requires_grad = False
            if module.bias is not None:
                merge_layer.bias.data.copy_(module.bias.data)

            # merge_layer.eval_init()
            # Replace the layer in merged_model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = merged_model.get_submodule(parent_name)
                setattr(parent, child_name, merge_layer)
            else:
                setattr(merged_model, child_name, merge_layer)
    del model1
    return merged_model

def convert(model1):
    import copy
    # Create a new model by copying model1's structure
    merged_model = copy.deepcopy(model1)
    # time = Timer()
    # Iterate through all modules in the model
    for name, module in merged_model.named_modules():
        # Replace Linear layers with MergeLinear layers
        if isinstance(module, MergeLinear) and "lm_head" not in name:
            # Create new MergeLinear with same dimensions
            q_layer = QLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )

            # Copy weights from model1 to main weights
            weight, scales = module.get_weights()
            scales = scales.squeeze(1)
            q_layer.scales.data.copy_(scales)
            q_layer.weight.data.copy_(weight)
            if module.bias is not None:
                q_layer.bias.data.copy_(module.bias.data)

            # merge_layer.eval_init()
            # Replace the layer in merged_model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = merged_model.get_submodule(parent_name)
                setattr(parent, child_name, q_layer)
            else:
                setattr(merged_model, child_name, q_layer)
    del model1
    return merged_model

def get_model(model1n):
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model1n,
            device_map="cuda",
            torch_dtype = torch.bfloat16
        )
    model.type(torch.bfloat16)
    # print(student_model)
    with torch.no_grad():
            # TODO: use a different method which is better supported, an official third party library
       model = merge(model)
       # student_model.model_tags = ["bitnet", "1.58b"]
    #loaded = load_file(path+"/model.safetensors")
    #student_model.load_state_dict(loaded)
    print("merged layers initialized. loading weights")
    load_sharded_checkpoint(model, model1n, strict = False)
    model.type(torch.bfloat16)
    model = model.cuda()
    model = convert(model)
    print("in evaluation mode. weights have been calculated")
    # print(student_model.parameters())
    model.type(torch.bfloat16)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model using lm-eval harness.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--push_path",
        default="kaizen9/pretrain_4000steps_200warm__20B_200k_FP",
        action="store_true",
        help="Use lossless compression.",
    )
    args = parser.parse_args()
    t = get_model(args.checkpoint_path)
    t.push_to_hub(args.push_path)
