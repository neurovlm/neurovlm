"""Saving and loading models."""

from typing import Union
import torch
from torch import nn
from safetensors.torch import save_file, load_file

def save_model(model: Union[str, nn.Module], out_file: str, device="cpu"):
    """Save model to .safetensors file."""
    if isinstance(model, str):
        model = torch.load(model, map_location=device) 
    save_file(model.state_dict(), out_file)

def load_model(model: nn.Module, safetensors_path: str, eval=True, device="cpu"):
    """Load model from .safetensors file."""
    state = load_file(safetensors_path, device="cpu") 
    model.load_state_dict(state, strict=True)
    if eval:
        model.eval()
    if device != "cpu":
        model.to(device)
    return model
