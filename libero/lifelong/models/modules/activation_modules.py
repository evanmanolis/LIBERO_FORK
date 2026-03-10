from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class SafeRational(nn.Module):
    """Safe rational activation: P(x) / (1 + |Q(x)|).

    The default polynomial degrees are intentionally small to keep training stable.
    """

    def __init__(
        self,
        numerator_degree: int = 3,
        denominator_degree: int = 2,
        input_clip: Optional[float] = 10.0,
        init: str = "identity",
    ):
        super().__init__()
        if numerator_degree < 0 or denominator_degree < 0:
            raise ValueError("Polynomial degrees must be non-negative")
        if numerator_degree > 5 or denominator_degree > 5:
            raise ValueError(
                "Polynomial degrees are capped at 5 for numerical stability"
            )
        if input_clip is not None and input_clip <= 0:
            raise ValueError("input_clip must be positive or None")

        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        self.input_clip = input_clip
        self.init = init

        self.p = nn.Parameter(torch.zeros(numerator_degree + 1))
        self.q = nn.Parameter(torch.zeros(denominator_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.p.zero_()
            self.q.zero_()
            if self.init == "identity":
                if self.numerator_degree >= 1:
                    self.p[1] = 1.0
                else:
                    self.p[0] = 1.0
            elif self.init == "small_random":
                self.p.normal_(mean=0.0, std=0.01)
                self.q.normal_(mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unsupported SafeRational init '{self.init}'")

    @staticmethod
    def _poly(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Horner's method for stable polynomial evaluation.
        out = torch.zeros_like(x) + coeffs[-1]
        for c in reversed(coeffs[:-1]):
            out = out * x + c
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_clip is not None:
            x = torch.clamp(x, -self.input_clip, self.input_clip)
        numerator = self._poly(self.p, x)
        denominator = 1.0 + torch.abs(self._poly(self.q, x))
        return numerator / denominator


def get_activation(
    name: str,
    *,
    activation_kwargs: Optional[Dict[str, Any]] = None,
    inplace: bool = False,
) -> nn.Module:
    kwargs = dict(activation_kwargs or {})
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=kwargs.pop("inplace", inplace))
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=kwargs.pop("inplace", inplace))
    if name in {"identity", "none"}:
        return nn.Identity()
    if name in {"rational", "safe_rational", "safe-rational"}:
        return SafeRational(**kwargs)
    raise ValueError(f"Unsupported activation '{name}'")
