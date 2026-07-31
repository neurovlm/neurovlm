"""Adapters for text-to-brain generation."""

from typing import Optional, Iterable
import torch
from torch import nn


class InterleavedResidualBlock(nn.Module):
    """Extra trainable layers inserted at a fixed decoder hidden dimension.

    Starts as an identity function because the final projection is zero-initialized:

        output = x + scale * f(x)

    At initialization, f(x) = 0, so output = x.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.fc1(self.dropout(self.activation(self.fc0(self.norm(x)))))
        return x + self.scale * dx

class LogitCalibrator(nn.Module):
    """Optional final logit calibration.

    Learns a temperature and bias after the pretrained decoder output.
    """

    def __init__(
        self,
        dim_out: int,
        per_feature_bias: bool = True,
        learn_temperature: bool = True,
    ):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros((dim_out,) if per_feature_bias else (1,)))

        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_temperature", torch.zeros(()))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return logits / temperature + self.bias


class InterleavedDecoderAdapter(nn.Module):
    """Projection head + pretrained decoder with extra trainable blocks inserted between decoder layers.

    Expected pretrained decoder structure:

        Linear(dim_latent, dim_h1)
        Activation
        Linear(dim_h1, dim_h0)
        Activation
        Linear(dim_h0, dim_out)
        optional Sigmoid

    The inserted blocks are:

        latent -> dec0 -> act0 -> extra_h1 -> dec1 -> act1 -> extra_h0 -> dec2

    This keeps the pretrained decoder usable at initialization while adding trainable
    capacity between its hidden stages.
    """

    def __init__(
        self,
        pretrained: nn.Module,
        proj_head: nn.Module,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        use_output_calibrator: bool = True,
        output_per_feature_bias: bool = True,
        freeze_pretrained_decoder: bool = True,
        freeze_proj_head: bool = False,
        return_logits: bool = True,
    ):
        super().__init__()

        dec = list(pretrained.decoder)

        if isinstance(dec[-1], nn.Sigmoid):
            dec = dec[:-1]

        if len(dec) < 5:
            raise ValueError(
                "Expected pretrained.decoder to look like: "
                "Linear, Act, Linear, Act, Linear, optional Sigmoid."
            )

        self.proj_head = proj_head

        self.dec0 = dec[0]
        self.dec_act0 = dec[1]
        self.extra_h1 = InterleavedResidualBlock(
            dim=self.dec0.out_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.dec1 = dec[2]
        self.dec_act1 = dec[3]
        self.extra_h0 = InterleavedResidualBlock(
            dim=self.dec1.out_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.dec2 = dec[4]

        dim_out = self.dec2.out_features
        self.output_calibrator = (
            LogitCalibrator(
                dim_out=dim_out,
                per_feature_bias=output_per_feature_bias,
                learn_temperature=True,
            )
            if use_output_calibrator
            else nn.Identity()
        )

        self.return_logits = return_logits

        if freeze_pretrained_decoder:
            for module in [self.dec0, self.dec1, self.dec2]:
                for p in module.parameters():
                    p.requires_grad = False

        if freeze_proj_head:
            for p in self.proj_head.parameters():
                p.requires_grad = False

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_act0(self.dec0(z))
        x = self.extra_h1(x)

        x = self.dec_act1(self.dec1(x))
        x = self.extra_h0(x)

        logits = self.dec2(x)
        logits = self.output_calibrator(logits)

        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_head(x)
        logits = self.decode_logits(z)

        if self.return_logits:
            return logits

        return torch.sigmoid(logits)

    def inserted_parameters(self) -> Iterable[nn.Parameter]:
        for module in [self.extra_h1, self.extra_h0, self.output_calibrator]:
            yield from module.parameters()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from (p for p in self.parameters() if p.requires_grad)
