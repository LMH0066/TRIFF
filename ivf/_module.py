import copy
import math
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import settings
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import Tensor
from torch.nn.modules import Dropout, LayerNorm, Linear

from ivf._utils import LOSS_KEYS


class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()

        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.register_buffer("PE", pe)

    def forward(self, x):
        # [batch_size, seq_len]
        seq_len = x.size(1)
        return self.PE[:seq_len, :]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.4, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, targets, reduction="none"
        )

        prob = torch.sigmoid(outputs)
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class NET(BaseModuleClass):
    def __init__(
        self,
        main_loc: str,
        target_loc: str,
        output_dim: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        seed: int = 42,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

        self.main_loc, self.target_loc = main_loc, target_loc
        self.num_labels = output_dim

        self.embedding = nn.Linear(1, d_model)
        self.pos_embedding = PositionEncoding(100, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        encoder_norm = LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2, **factory_kwargs),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, **factory_kwargs),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=bias),
            LayerNorm(d_model // 2, eps=layer_norm_eps, bias=bias, **factory_kwargs),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4, bias=bias),
            LayerNorm(d_model // 4, eps=layer_norm_eps, bias=bias, **factory_kwargs),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(d_model // 4, output_dim, bias=bias),
        )

        self._init_weights()

    def _get_inference_input(self, tensors: Dict[Any, Any], **kwargs):
        main = tensors[self.main_loc]  # [batch_size, tokens]

        input_dict = {"main": main}
        return input_dict

    @auto_move_data
    def inference(self, main: torch.Tensor):
        # Keep batch and feature axes distinct, including one-feature ablations.
        _main = main
        if _main.ndim == 1:
            _main = _main.unsqueeze(0)
        elif _main.ndim == 3 and _main.shape[0] == 1:
            _main = _main.squeeze(0)
        if _main.ndim != 2:
            raise ValueError("Expected [batch, features] input")
        _x = torch.nan_to_num(_main, nan=-1.0)
        _x = self.embedding(_x.unsqueeze(-1)).permute(1, 0, 2) + self.pos_embedding(
            _x
        ).unsqueeze(1)
        attn_logits = self.encoder(_x, src_key_padding_mask=torch.isnan(_main))

        attention_weights = self.attention_pool(attn_logits)
        attention_weights = F.softmax(attention_weights, dim=0) # [features, batch_size, 1]
        pooled = (attn_logits * attention_weights).sum(dim=0) # [batch_size, d_model]

        out = self.decoder(pooled)
        return out

    @auto_move_data
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_output: torch.Tensor,
    ) -> Dict[str, float]:
        losses = dict()
        target = tensors[self.target_loc].squeeze()

        losses[LOSS_KEYS.FocalLoss] = FocalLoss()(inference_output.squeeze(), target)

        return losses

    @auto_move_data
    def forward(
        self,
        tensors,
        compute_loss=True,
        get_inference_input_kwargs: Optional[dict] = None,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, LossOutput]]:
        inference_inputs = self._get_inference_input(tensors)
        inference_output = self.inference(**inference_inputs)

        if compute_loss:
            losses = self.loss(tensors, inference_output)
            return inference_output, losses
        else:
            return inference_output

    def _init_weights(self, initializer_range: float = 0.02):
        """Initialize weights using proper initialization strategies."""
        for module in self.modules():
            # if isinstance(module, nn.Embedding):
            #     nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def load_state_dict(self, state_dict):
        def _remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix) :]
            return text

        pairings = [
            (src_key, _remove_prefix(src_key, "_orig_mod."))
            for src_key in state_dict.keys()
        ]
        if all(src_key == dest_key for src_key, dest_key in pairings):
            super(NET, self).load_state_dict(state_dict)
            return  # Do not write checkpoint if no need to repair!

        _state_dict = {}
        for src_key, dest_key in pairings:
            _state_dict[dest_key] = state_dict[src_key]
        super(NET, self).load_state_dict(_state_dict)
