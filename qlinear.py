import torch
from torch import nn
import torch.nn.functional as F

def modified_weight_quant(w):
    """ Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    # modified via different scale multiplication, and no internal scaling factor
    # note that the quantized weights, post PTQ should be in the integer/scaling factor format
    # m = w.abs().mean() * 0.8
    # scale = 1.0 / m.clamp_(min=1e-5) # original has it be based off mean, but since we initialize differently, we change it
    # u = (w * scale).round().clamp_(-1, 1)
    u = w.clamp(-1, 1).round() 
    return u

def normalize(w):
    w = w / torch.norm(w, dim=1, keepdim=True) 
    return w

class QLinear(nn.Linear):
    def __init__(self,
            *kargs,
            **kwargs
        ):
        super(QLinear, self).__init__(*kargs, **kwargs)
        """
        This is only for training, and kernel optimization is needed for efficiency.
        """
        self.scales = nn.Parameter(torch.ones(self.out_features))
        self.quantizer = modified_weight_quant
        self.weight.register_hook(self.weight_hook)


    def forward(self, x):
        """i
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        w_quant = self.weight
        x = x.to(w_quant.device)
        # STE weight quantization
        w_quant = w_quant + (self.quantizer(w_quant) - w_quant).detach()
        w_quant = normalize(w_quant)
        y = F.linear(x, w_quant) 
        # apply scales post matmul
        y = y * self.scales
        if self.bias is not None:
            y = y + self.bias
        return y
    def weight_hook(self, grad_output):
        grad = grad_output.clone()  # Preserve original gradients
    
        input = self.weight.clone().detach()
        ternary_values = input.round()  # -1, 0, or +1
        scale = torch.abs(input - ternary_values)  # Uncertainty measure (0 = certain, 0.5 = maximally uncertain)
    
        # 1. Scale gradients by uncertainty (prioritize uncertain weights)
        grad = grad * scale
    
        # 2. Adaptive thresholding (optional but recommended)
        # if grad.numel() > 1:  # Only apply if there are multiple elements
        #     threshold = 3 * grad.std()
        #     grad = grad * (grad.abs() >= threshold)
    
        # 3. Gradient clipping to prevent explosions (critical after scaling)
        # max_value, _ = torch.max(grad)
        # if max_value > 1:
        #     max_value = max_value * 0.7
        #     grad = grad * (1 / max_value)
        # Normalize grad if norm > 1
        grad_norm = torch.norm(grad)
        if grad_norm > 1:
            grad = grad / grad_norm

    
        return grad
    

