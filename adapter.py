import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn


class MicroAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.scaling = nn.Parameter(torch.tensor(alpha / max(1, rank), dtype=torch.float32))
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        delta = self.up(self.down(self.dropout(x))) * self.scaling
        return delta.to(orig_dtype)


class DeepAdapterWrapper(nn.Module):
    """
    REInE wrapper with three useful features for training:
    1. residual adapters on selected layers
    2. per-layer static scale parameters
    3. hidden-state capture for latent anchor loss on lower layers

    Chat/inference can stay unchanged. Training can optionally read pooled
    hidden states from self.hidden_cache after a forward pass.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer=None,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.05,
        target_layers: Optional[List[int]] = None,
        layer_scale_init: Optional[Dict[int, float]] = None,
        capture_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.base = base_model
        self.tokenizer = tokenizer
        self.adapters_enabled = True
        self.capture_enabled = True
        self.hidden_cache: Dict[str, torch.Tensor] = {}

        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, "n_embd")
        self.hidden_size = hidden_size

        if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            self.blocks = base_model.transformer.h
        elif hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            self.blocks = base_model.model.layers
        else:
            raise RuntimeError("Unsupported model architecture")

        num_layers = len(self.blocks)
        if target_layers is None:
            self.target_layers = list(range(num_layers))
        else:
            self.target_layers = sorted(idx for idx in target_layers if 0 <= idx < num_layers)
        if not self.target_layers:
            raise ValueError("No valid target_layers were provided.")

        self.capture_layers = sorted(set(capture_layers or self.target_layers))

        self.adapters = nn.ModuleDict({
            str(idx): MicroAdapter(hidden_size, rank=rank, alpha=alpha, dropout=dropout)
            for idx in self.target_layers
        })

        layer_scale_init = layer_scale_init or {}
        self.layer_scales = nn.ParameterDict({
            str(idx): nn.Parameter(torch.tensor(float(layer_scale_init.get(idx, 1.0)), dtype=torch.float32))
            for idx in self.target_layers
        })

        self._register_hooks()

    def clear_hidden_cache(self):
        self.hidden_cache = {}

    def set_adapters_enabled(self, enabled: bool):
        self.adapters_enabled = enabled

    def set_capture_enabled(self, enabled: bool):
        self.capture_enabled = enabled

    def freeze_layer_scales(self):
        for p in self.layer_scales.parameters():
            p.requires_grad = False

    @staticmethod
    def masked_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        hidden: [B, T, H]
        attention_mask: [B, T]
        returns: [B, H]
        """
        if attention_mask is None:
            return hidden.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom

    def get_pooled_hidden(self, layer_idx: int, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        key = str(layer_idx)
        if key not in self.hidden_cache:
            raise KeyError(f"Layer {layer_idx} not found in hidden_cache. capture_layers={self.capture_layers}")
        return self.masked_mean(self.hidden_cache[key], attention_mask)

    def _register_hooks(self):
        for idx, block in enumerate(self.blocks):
            needs_adapter = idx in self.target_layers
            needs_capture = idx in self.capture_layers
            if not needs_adapter and not needs_capture:
                continue

            adapter = self.adapters[str(idx)] if needs_adapter else None
            layer_scale = self.layer_scales[str(idx)] if needs_adapter else None

            def make_hook(layer_idx, adapter_module, layer_scale_param):
                def hook(module, inputs, output):
                    is_tuple = isinstance(output, tuple)
                    h = output[0] if is_tuple else output

                    if self.adapters_enabled and adapter_module is not None:
                        delta = adapter_module(h)
                        h = h + layer_scale_param.to(h.dtype) * delta

                    if self.capture_enabled and layer_idx in self.capture_layers:
                        self.hidden_cache[str(layer_idx)] = h

                    return (h,) + output[1:] if is_tuple else h
                return hook

            block.register_forward_hook(make_hook(idx, adapter, layer_scale))

    def forward(self, *args, **kwargs):
        self.clear_hidden_cache()
        return self.base(*args, **kwargs)
