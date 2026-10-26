# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
perception.py -- Perception stage of Aero-VPA (preprint_261026.pdf, Sec. 3.2, Eq. 6, Table 1).

Implements
    * ``RGBDNormalizer``     -- parameter-free normalisation of one RGB-D frame (Sec. 3.2): RGB by
                                per-channel mean/std, depth clipped to the sensor range and scaled to
                                [0, 1]. It also returns the clipped metric depth and the validity mask
                                that the braking envelope (Sec. 3.6) needs; the envelope must never read
                                the normalised network channel.
    * ``VisionEncoder``      -- E_psi, the RGB-D Vision Transformer (Eq. 6):
                                    z_t = E_psi(I_t) in R^d,   I_t in R^{4 x 128 x 160}
                                patch 16 -> N_p = (128/16)(160/16) = 80 tokens plus [CLS], width 384,
                                6 pre-norm blocks with 6 heads and MLP ratio 4, final LayerNorm, and
                                Linear(384, 256) on the [CLS] token: 11,171,200 parameters (Table 1).
    * ``MomentumEncoder``    -- E_psi_bar, the EMA copy that produces the stop-gradient JEPA target
                                sg(E_psi_bar(I_{t+1})) of Eq. 8.
    * ``TextEncoderWrapper`` -- the frozen CLIP text tower (c_text in R^512, computed once per mission)
                                and the learned null embedding used when no instruction is given.

Changes from VPA (preprint_261008)
    * Frames are 4-channel RGB-D at 128 x 160 (was 3 x 224 x 224), so N_p = 80 (was 196).
    * Aero-VPA waypoints are geometric (Eq. 14), so milestone-frame encoding (``encode_milestones``,
      ``encode_siamese``) is removed, and so is the multi-camera path: the vehicle has one front RGB-D
      camera.
    * The final LayerNorm is applied to the [CLS] token only (LayerNorm is per token, so this is the
      same map with 1/81 of the work).
    * ``TextEncoderWrapper`` gains the learned null embedding (its only trainable parameter).

Layout convention
    The paper writes I_t in R^{H_img x W_img x 4}. Tensors here are channels-first,
    ``[B, 4, 128, 160]`` with channels (R, G, B, D). Patch tokens are row-major: token n covers
    patch row n // 10 and patch column n % 10.

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python perception.py``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "AERO_IMAGE_SIZE",
    "AERO_IN_CHANNELS",
    "AERO_ENCODER_PARAMS",
    "count_parameters",
    "RGBDInput",
    "RGBDNormalizer",
    "PatchEmbedding",
    "TransformerBlock",
    "VisionEncoder",
    "MomentumEncoder",
    "TextEncoderWrapper",
]

AERO_IMAGE_SIZE: Tuple[int, int] = (128, 160)  # (H_img, W_img), Sec. 3.1
AERO_IN_CHANNELS: int = 4                      # R, G, B, D
AERO_ENCODER_PARAMS: int = 11_171_200          # Table 1: 11.17M

ImageSize = Union[int, Tuple[int, int]]


def _pair(x: ImageSize) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else (int(x[0]), int(x[1]))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters of ``module``; only those with ``requires_grad`` if ``trainable_only``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------------------------
# RGB-D input normalisation (Sec. 3.2)
# ---------------------------------------------------------------------------------------------
class RGBDInput(NamedTuple):
    """Output of ``RGBDNormalizer``."""

    image: Tensor    # [B, 4, H, W]  network input I_t: normalised R, G, B and D
    depth_m: Tensor  # [B, H, W]     metric depth (m) clipped to the sensor range; 0 where invalid
    valid: Tensor    # [B, H, W]     bool, True where the depth reading is usable


class RGBDNormalizer(nn.Module):
    """Normalise one RGB-D frame for E_psi (Sec. 3.2). Parameter-free (two constant buffers).

    RGB: uint8 frames are scaled to [0, 1]; float frames must already be in [0, 1]. Then
    (x - mean) / std per channel.

    Depth: a reading is valid if it is finite and > 0 (0 is the usual no-data value of stereo
    sensors). Valid readings are clipped to [d_min, d_max] and the network channel is d / d_max, in
    [d_min / d_max, 1]. Invalid pixels are 0 in the network channel and 0 m in ``depth_m``, so code
    that forgets the mask is conservative: it sees an obstacle at zero range, never free space.

    The paper fixes only "clipped to the sensor range and normalised". The statistics and the
    invalid-pixel convention are implementation choices.

    Args:
        depth_range: (d_min, d_max) of the depth sensor in metres, 0 < d_min < d_max.
        rgb_mean, rgb_std: per-channel statistics on the [0, 1] scale (ImageNet by default).

    Shapes:
        ``forward``: (``rgb [B, 3, H, W]`` uint8 or float, ``depth [B, H, W]`` or ``[B, 1, H, W]`` in m)
                     -> ``RGBDInput(image [B, 4, H, W], depth_m [B, H, W], valid [B, H, W] bool)``
    """

    def __init__(
        self,
        depth_range: Tuple[float, float] = (0.1, 10.0),
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        d_min, d_max = float(depth_range[0]), float(depth_range[1])
        if not 0.0 < d_min < d_max:
            raise ValueError(f"depth_range must satisfy 0 < d_min < d_max, got {(d_min, d_max)}")
        if len(rgb_mean) != 3 or len(rgb_std) != 3:
            raise ValueError("rgb_mean and rgb_std need one value per RGB channel")
        if min(float(s) for s in rgb_std) <= 0.0:
            raise ValueError("rgb_std must be strictly positive")
        self.depth_min: float = d_min
        self.depth_max: float = d_max
        self.register_buffer("rgb_mean", torch.tensor([float(m) for m in rgb_mean], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([float(s) for s in rgb_std], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, rgb: Tensor, depth: Tensor) -> RGBDInput:
        """Build I_t ``[B, 4, H, W]`` (R, G, B, D) plus metric depth and its validity mask."""
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected rgb [B, 3, H, W], got {tuple(rgb.shape)}")
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        b, _, h, w = rgb.shape
        if tuple(depth.shape) != (b, h, w):
            raise ValueError(f"expected depth [{b}, {h}, {w}] or [{b}, 1, {h}, {w}], got {tuple(depth.shape)}")
        x = rgb.to(self.rgb_mean.dtype)
        if rgb.dtype == torch.uint8:
            x = x / 255.0
        x = (x - self.rgb_mean) / self.rgb_std                                       # [B, 3, H, W]
        d = depth.to(self.rgb_mean.dtype)
        valid = torch.isfinite(d) & (d > 0.0)                                        # [B, H, W]
        depth_m = torch.where(valid, d.clamp(self.depth_min, self.depth_max), torch.zeros_like(d))
        image = torch.cat([x, (depth_m / self.depth_max).unsqueeze(1)], dim=1)       # [B, 4, H, W]
        return RGBDInput(image, depth_m, valid)


# ---------------------------------------------------------------------------------------------
# Vision Transformer building blocks
# ---------------------------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Non-overlapping patch tokenizer of a Vision Transformer.

    Args:
        image_size: (H_img, W_img) or a single int for square frames.
        patch_size: side length P of the square patches; must divide H_img and W_img.
        in_channels: number of image channels C.
        width: token width D_vit.

    Shapes:
        input  ``images``: ``[B, C, H_img, W_img]``
        output ``tokens``: ``[B, N_patch, D_vit]``, N_patch = (H_img / P)(W_img / P), row-major order
    """

    def __init__(self, image_size: ImageSize, patch_size: int, in_channels: int, width: int) -> None:
        super().__init__()
        h, w = _pair(image_size)
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"image size {(h, w)} is not divisible by patch size {patch_size}")
        self.image_size: Tuple[int, int] = (h, w)
        self.patch_size: int = patch_size
        self.grid_size: Tuple[int, int] = (h // patch_size, w // patch_size)
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> Tensor:
        """``[B, C, H_img, W_img]`` -> ``[B, N_patch, D_vit]``."""
        x = self.proj(images)                 # [B, D_vit, H_img/P, W_img/P]
        return x.flatten(2).transpose(1, 2)   # [B, N_patch, D_vit]


class TransformerBlock(nn.Module):
    """Pre-norm ViT block: x <- x + MHSA(LN(x));  x <- x + MLP(LN(x)).

    Only LayerNorm is used (no BatchNorm), so every sample in a batch is processed independently.

    Shapes:
        input/output ``x``: ``[B, N_tok, D_vit]``
    """

    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width {width} must be divisible by num_heads {num_heads}")
        hidden = int(round(width * mlp_ratio))
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N_tok, D_vit]`` -> ``[B, N_tok, D_vit]``."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------------------------
# E_psi : RGB-D vision encoder (Eq. 6)
# ---------------------------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    """E_psi, the RGB-D Vision Transformer of Aero-VPA (Sec. 3.2, Eq. 6, Table 1).

        z_t = E_psi(I_t) = W_head LN(ViT(I_t))_[CLS] + b_head  in R^d

    The defaults are the Table 1 configuration, 11,171,200 parameters:

        Conv2d(4, 384, k=16, s=16)            393,600   patch tokens [B, 80, 384]
        [CLS] token [1, 1, 384]                    384
        learned positions [1, 81, 384]          31,104
        6 x pre-norm block                  10,646,784   (1,774,464 each)
        final LayerNorm(384)                       768
        Linear(384, 256) head                   98,560

    Args:
        image_size: (H_img, W_img) or int.
        patch_size: ViT patch size P.
        in_channels: image channels C (4 for RGB-D).
        width: ViT token width D_vit.
        depth: number of transformer blocks.
        num_heads: attention heads per block.
        latent_dim: dimension d of the latent space Z.
        mlp_ratio: hidden width multiplier of the block MLPs.

    Shapes (``frame_shape`` = ``[C, H_img, W_img]`` = ``[4, 128, 160]`` by default):
        ``tokens``:   ``[B, *frame_shape]`` -> ``[B, N_patch + 1, D_vit]`` ([CLS] first, positions added)
        ``features``: ``[B, *frame_shape]`` -> ``[B, D_vit]``  (final-LayerNorm [CLS] token)
        ``forward``:  ``[B, *frame_shape]`` -> ``[B, d]``
    """

    def __init__(
        self,
        image_size: ImageSize = AERO_IMAGE_SIZE,
        patch_size: int = 16,
        in_channels: int = AERO_IN_CHANNELS,
        width: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        latent_dim: int = 256,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.in_channels: int = int(in_channels)
        self.width: int = int(width)
        self.latent_dim: int = int(latent_dim)
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, width)
        self.image_size: Tuple[int, int] = self.patch_embed.image_size
        self.num_patches: int = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Shape of one observation, ``(C, H_img, W_img)``."""
        return (self.in_channels, *self.image_size)

    def _check_frames(self, images: Tensor) -> None:
        if images.dim() != 4 or tuple(images.shape[1:]) != self.frame_shape:
            c, h, w = self.frame_shape
            raise ValueError(f"expected frames [B, {c}, {h}, {w}], got {tuple(images.shape)}")

    def tokens(self, images: Tensor) -> Tensor:
        """Input tokens of the first block: ``[B, C, H, W]`` -> ``[B, N_patch + 1, D_vit]``."""
        self._check_frames(images)
        x = self.patch_embed(images)                          # [B, N_patch, D_vit]
        cls = self.cls_token.expand(x.shape[0], -1, -1)       # [B, 1, D_vit]
        return torch.cat([cls, x], dim=1) + self.pos_embed    # [B, N_patch + 1, D_vit]

    def features(self, images: Tensor) -> Tensor:
        """Final-LayerNorm [CLS] token: ``[B, C, H, W]`` -> ``[B, D_vit]``.

        A plain method, not a submodule, so ``forward`` is one evaluation of E_psi for the
        sequential-depth audit of the pipeline.
        """
        x = self.tokens(images)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])                             # [B, D_vit]

    def forward(self, images: Tensor) -> Tensor:
        """Eq. 6: z = E_psi(I), ``[B, 4, 128, 160]`` -> ``[B, d]``."""
        return self.head(self.features(images))


# ---------------------------------------------------------------------------------------------
# E_psi_bar : momentum target encoder (Eq. 8)
# ---------------------------------------------------------------------------------------------
class MomentumEncoder(nn.Module):
    """Momentum (EMA) copy E_psi_bar of the online encoder, used only to produce JEPA targets.

    Update after each optimiser step: psi_bar <- mu psi_bar + (1 - mu) psi, mu in (0, 1) (Sec. 3.2).
    The target branch receives no gradients: ``forward`` runs under ``torch.no_grad`` and returns a
    detached tensor, which realises the stop-gradient sg(.) of Eq. 8. Its parameters do not count
    towards the trainable budget of Table 1.

    Shapes:
        ``forward``: ``[B, *frame_shape]`` -> ``[B, d]`` (detached)
    """

    def __init__(self, online_encoder: VisionEncoder, momentum: float = 0.996) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        self.momentum: float = float(momentum)
        self.encoder: VisionEncoder = copy.deepcopy(online_encoder)
        self.encoder.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: VisionEncoder) -> None:
        """psi_bar <- momentum * psi_bar + (1 - momentum) * psi (parameters; buffers are copied)."""
        for p_bar, p in zip(self.encoder.parameters(), online_encoder.parameters(), strict=True):
            p_bar.mul_(self.momentum).add_(p.detach(), alpha=1.0 - self.momentum)
        for b_bar, b in zip(self.encoder.buffers(), online_encoder.buffers(), strict=True):
            b_bar.copy_(b)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """``[B, *frame_shape]`` -> ``[B, d]`` target latents sg(E_psi_bar(I))."""
        return self.encoder(images).detach()


# ---------------------------------------------------------------------------------------------
# Frozen text encoder and learned null embedding (c_text)
# ---------------------------------------------------------------------------------------------
Instructions = Union[None, str, Sequence[Optional[str]]]


class TextEncoderWrapper(nn.Module):
    """Instruction embedding c_text in R^{d_c} (Sec. 3.2).

    With an instruction x, c_text is the projected pooled output (``text_embeds``) of a frozen,
    pre-trained CLIP text tower (``transformers.CLIPTextModelWithProjection`` + ``CLIPTokenizer``),
    so d_c = ``config.projection_dim`` = 512 for ``openai/clip-vit-base-patch32`` (63.4M frozen
    parameters, evaluated once per mission, outside the per-decision chain). Without an instruction
    (``None`` or an empty string), c_text is the learned null embedding ``null_embedding`` in R^{d_c},
    the wrapper's only trainable parameter (trained in stage (ii), Sec. 3.8). The CLIP tower stays in
    eval mode permanently.

    For offline use or tests, a ``text_model`` and ``tokenizer`` can be injected. The injected model
    must accept ``(input_ids, attention_mask)``, return an object with a ``text_embeds`` attribute of
    shape ``[B, d_c]`` and expose ``config.projection_dim``.

    Shapes:
        ``forward``:       None, one string, or B entries (str or None) -> ``[B, d_c]`` (B = 1 for a single entry)
        ``encode_tokens``: ``input_ids [B, L]`` (+ ``attention_mask [B, L]``) -> ``[B, d_c]``
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        text_model: Optional[nn.Module] = None,
        tokenizer: Optional[Callable[..., Any]] = None,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        if text_model is None or tokenizer is None:
            # Imported lazily so the rest of this module works without `transformers` installed.
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer

            if text_model is None:
                text_model = CLIPTextModelWithProjection.from_pretrained(model_name_or_path)
            if tokenizer is None:
                tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
        self.text_model: nn.Module = text_model
        self.tokenizer: Callable[..., Any] = tokenizer
        self.max_length: int = int(max_length)
        self.embed_dim: int = int(text_model.config.projection_dim)  # d_c
        self.text_model.requires_grad_(False)
        self.text_model.eval()
        self.null_embedding = nn.Parameter(torch.zeros(self.embed_dim))  # learned c_text without instruction

    def train(self, mode: bool = True) -> "TextEncoderWrapper":
        """The CLIP tower is frozen: it stays in eval mode regardless of ``mode``."""
        super().train(mode)
        self.text_model.eval()
        return self

    def _device(self) -> torch.device:
        try:
            return next(self.text_model.parameters()).device
        except StopIteration:
            return self.null_embedding.device

    @staticmethod
    def _is_null(instruction: Optional[str]) -> bool:
        return instruction is None or instruction.strip() == ""

    @torch.no_grad()
    def encode_tokens(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Embed pre-tokenised instructions with the frozen tower.

        Args:
            input_ids:      ``[B, L]`` int64 token ids.
            attention_mask: ``[B, L]`` (optional) 1 for real tokens, 0 for padding.

        Returns:
            c_text: ``[B, d_c]`` (no gradient).
        """
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        c_text = out.text_embeds if hasattr(out, "text_embeds") else out
        if c_text.dim() != 2 or c_text.shape[-1] != self.embed_dim:
            raise RuntimeError(f"text model returned {tuple(c_text.shape)}, expected [B, {self.embed_dim}]")
        return c_text

    def forward(self, instructions: Instructions) -> Tensor:
        """c_text for each entry: CLIP for an instruction, the learned null embedding for None / "".

        Rows from CLIP carry no gradient; null rows are differentiable with respect to
        ``null_embedding``.

        Args:
            instructions: None, one string, or a sequence of B entries (strings or None).

        Returns:
            c_text: ``[B, d_c]`` (B = 1 for None or a single string).
        """
        items = [instructions] if instructions is None or isinstance(instructions, str) else list(instructions)
        if len(items) == 0:
            raise ValueError("at least one instruction (or None) is required")
        is_text = [not self._is_null(x) for x in items]
        null = self.null_embedding.unsqueeze(0).expand(len(items), -1)                 # [B, d_c]
        if not any(is_text):
            return null
        texts = [x for x, keep in zip(items, is_text, strict=True) if keep]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        device = self._device()
        attention_mask = tokens.get("attention_mask")
        encoded = self.encode_tokens(
            tokens["input_ids"].to(device),
            attention_mask.to(device) if attention_mask is not None else None,
        )                                                                               # [B_text, d_c]
        if all(is_text):
            return encoded
        index = torch.tensor([i for i, keep in enumerate(is_text) if keep], device=encoded.device)
        rows = encoded.new_zeros(len(items), self.embed_dim).index_copy(0, index, encoded)
        keep = torch.tensor(is_text, device=encoded.device).unsqueeze(1)                # [B, 1]
        return torch.where(keep, rows, null.to(device=encoded.device, dtype=encoded.dtype))


# ---------------------------------------------------------------------------------------------
# Self-test (run: python perception.py)
# ---------------------------------------------------------------------------------------------
class _ToyTextModel(nn.Module):
    """Tiny stand-in with the CLIPTextModelWithProjection interface (offline testing only)."""

    def __init__(self, vocab: int = 100, width: int = 32, projection_dim: int = 24) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=projection_dim)
        self.embed = nn.Embedding(vocab, width)
        self.proj = nn.Linear(width, projection_dim, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> SimpleNamespace:
        h = self.embed(input_ids)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)
        return SimpleNamespace(text_embeds=self.proj(h.mean(1)))


def _toy_tokenizer(texts: Sequence[str], max_length: int = 16, **_: Any) -> dict:
    ids = torch.zeros(len(texts), max_length, dtype=torch.long)
    mask = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        codes = [ord(ch) % 100 for ch in text][:max_length]
        ids[i, : len(codes)] = torch.tensor(codes, dtype=torch.long)
        mask[i, : len(codes)] = 1
    return {"input_ids": ids, "attention_mask": mask}


_RTOL, _ATOL = 1e-5, 1e-6              # float32 module output vs float64 reference (CLAUDE.md)
_RTOL_WIDE, _ATOL_WIDE = 1e-4, 1e-5    # full Table 1 width: 6 blocks of 384 accumulate more float32 rounding

# Table 1 encoder budget by top-level submodule.
_ENCODER_BREAKDOWN = {
    "patch_embed": 393_600,    # Conv2d(4, 384, 16, 16): 4 * 16 * 16 * 384 + 384
    "cls_token": 384,
    "pos_embed": 31_104,       # (80 + 1) * 384
    "blocks": 10_646_784,      # 6 * 1,774,464
    "norm": 768,
    "head": 98_560,            # 384 * 256 + 256
}


def _close(actual: Tensor, ref: Tensor, rtol: float = _RTOL, atol: float = _ATOL) -> bool:
    return torch.allclose(actual.double(), ref.double(), rtol=rtol, atol=atol)


def _expect_value_error(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _ref_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """LayerNorm over the last axis, written out: (x - mean) / sqrt(biased var + eps) * w + b."""
    mean = x.sum(-1, keepdim=True) / x.shape[-1]
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / x.shape[-1]
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def _ref_vit_forward(enc: VisionEncoder, images: Tensor) -> Tensor:
    """Independent float64 reference of z = E_psi(I) (Eq. 6) for frames ``[B, C, H, W]``.

    Re-derives the ViT forward from its definition with float64 copies of the module weights:
    an explicit row-major patch loop, per-head softmax attention, exact erf-GELU, hand-written
    LayerNorm over all tokens, then the head on the [CLS] token. Returns ``[B, d]`` float64.
    """
    f64 = {k: v.detach().double() for k, v in enc.state_dict().items()}
    x = images.double()
    bsz, _, h, w = x.shape
    p = enc.patch_embed.patch_size
    w_patch = f64["patch_embed.proj.weight"].reshape(f64["patch_embed.proj.weight"].shape[0], -1)
    tokens = []
    for i in range(h // p):            # row-major patch order: rows outer, columns inner
        for j in range(w // p):
            patch = x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(bsz, -1)  # [B, C*P*P]
            tokens.append(patch @ w_patch.T + f64["patch_embed.proj.bias"])
    t = torch.stack(tokens, dim=1)     # [B, N_patch, D]
    t = torch.cat([f64["cls_token"].expand(bsz, -1, -1), t], dim=1) + f64["pos_embed"]
    width = t.shape[-1]
    for n, block in enumerate(enc.blocks):
        pre = f"blocks.{n}."
        heads = block.attn.num_heads
        dh = width // heads
        hn = _ref_layer_norm(t, f64[pre + "norm1.weight"], f64[pre + "norm1.bias"], block.norm1.eps)
        w_in, b_in = f64[pre + "attn.in_proj_weight"], f64[pre + "attn.in_proj_bias"]
        q = hn @ w_in[:width].T + b_in[:width]
        k = hn @ w_in[width:2 * width].T + b_in[width:2 * width]
        v = hn @ w_in[2 * width:].T + b_in[2 * width:]
        head_out = []
        for a in range(heads):
            sl = slice(a * dh, (a + 1) * dh)
            scores = q[..., sl] @ k[..., sl].transpose(1, 2) / dh ** 0.5  # [B, N, N]
            e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
            head_out.append((e / e.sum(dim=-1, keepdim=True)) @ v[..., sl])
        attn = torch.cat(head_out, dim=-1) @ f64[pre + "attn.out_proj.weight"].T + f64[pre + "attn.out_proj.bias"]
        t = t + attn
        hn = _ref_layer_norm(t, f64[pre + "norm2.weight"], f64[pre + "norm2.bias"], block.norm2.eps)
        hid = hn @ f64[pre + "mlp.0.weight"].T + f64[pre + "mlp.0.bias"]
        hid = 0.5 * hid * (1.0 + torch.erf(hid / 2.0 ** 0.5))  # exact GELU
        t = t + hid @ f64[pre + "mlp.2.weight"].T + f64[pre + "mlp.2.bias"]
    t = _ref_layer_norm(t, f64["norm.weight"], f64["norm.bias"], enc.norm.eps)
    return t[:, 0] @ f64["head.weight"].T + f64["head.bias"]  # [B, d]


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    b, c, (h, w), d = 2, AERO_IN_CHANNELS, AERO_IMAGE_SIZE, 256

    # ---- Table 1: the default encoder is the Aero-VPA E_psi with exactly 11,171,200 parameters.
    enc = VisionEncoder().eval()
    assert enc.frame_shape == (4, 128, 160)
    assert enc.patch_embed.grid_size == (8, 10) and enc.num_patches == 80 == (128 // 16) * (160 // 16)
    assert count_parameters(enc) == count_parameters(enc, trainable_only=False) == AERO_ENCODER_PARAMS
    assert round(AERO_ENCODER_PARAMS / 1e6, 2) == 11.17
    groups: dict = {}
    for name, p in enc.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    assert groups == _ENCODER_BREAKDOWN, groups
    assert len(enc.blocks) == 6 and all(count_parameters(blk) == 1_774_464 for blk in enc.blocks)
    assert all(blk.attn.num_heads == 6 for blk in enc.blocks)
    assert tuple(enc.patch_embed.proj.weight.shape) == (384, 4, 16, 16)
    assert tuple(enc.pos_embed.shape) == (1, 81, 384) and tuple(enc.head.weight.shape) == (256, 384)

    # ---- Eq. 6 tensor shapes: I_t [B, 4, 128, 160] -> patches [B, 80, 384] -> tokens [B, 81, 384]
    #      -> [CLS] [B, 384] -> z_t [B, 256]; every block sees [B, 81, 384].
    obs = torch.randn(b, c, h, w, generator=gen)
    seen = []
    hooks = [blk.register_forward_hook(lambda _m, args, _out: seen.append(tuple(args[0].shape))) for blk in enc.blocks]
    with torch.no_grad():
        patches = enc.patch_embed(obs)
        tok = enc.tokens(obs)
        feats = enc.features(obs)
        z_t = enc(obs)
    for hook in hooks:
        hook.remove()
    assert patches.shape == (b, 80, 384) and tok.shape == (b, 81, 384) and feats.shape == (b, 384)
    assert z_t.shape == (b, d) and z_t.dtype == torch.float32
    assert len(seen) == 2 * 6 and all(s == (b, 81, 384) for s in seen)
    # token 1 + n is patch (n // 10, n % 10): check one patch of the last row against the conv definition
    n = 75
    r, col = divmod(n, 10)
    patch = obs[:, :, 16 * r:16 * (r + 1), 16 * col:16 * (col + 1)].reshape(b, -1).double()
    w_patch = enc.patch_embed.proj.weight.detach().reshape(384, -1).double()
    assert _close(patches[:, n], patch @ w_patch.T + enc.patch_embed.proj.bias.detach().double())

    # ---- Eq. 6 against the independent float64 reference, at full Table 1 size and on a small
    #      non-square encoder (4 x 6 patch grid) at the repo tolerance.
    ref = _ref_vit_forward(enc, obs)
    assert _close(z_t, ref, _RTOL_WIDE, _ATOL_WIDE)
    with torch.no_grad():
        z_one = enc(obs[1:2])
    assert _close(z_one, ref[1:2], _RTOL_WIDE, _ATOL_WIDE)                    # B = 1: no cross-sample op
    small = VisionEncoder(image_size=(32, 48), patch_size=8, in_channels=c, width=32, depth=2,
                          num_heads=4, latent_dim=16).eval()
    x_small = torch.randn(3, c, 32, 48, generator=gen)
    with torch.no_grad():
        z_small = small(x_small)
    assert small.num_patches == 24 and z_small.shape == (3, 16)
    assert _close(z_small, _ref_vit_forward(small, x_small))

    # ---- Malformed frames and configurations raise ValueError.
    for shape in ((b, 3, h, w), (b, 5, h, w), (b, c, 224, 224), (b, c, w, h), (c, h, w), (b, 1, c, h, w)):
        _expect_value_error(lambda s=shape: enc(torch.zeros(*s)))
    _expect_value_error(lambda: VisionEncoder(image_size=(130, 160)))       # not divisible by 16
    _expect_value_error(lambda: VisionEncoder(num_heads=5))                 # 384 % 5 != 0

    # ---- RGB-D normaliser (Sec. 3.2): channel order (R, G, B, D), clipping, invalid pixels.
    norm = RGBDNormalizer(depth_range=(0.2, 8.0))
    assert count_parameters(norm, trainable_only=False) == 0
    rgb8 = torch.randint(0, 256, (b, 3, h, w), generator=gen, dtype=torch.uint8)
    depth = torch.rand(b, h, w, generator=gen) * 12.0                     # some readings beyond d_max
    depth[0, 0, :4] = torch.tensor([0.0, float("nan"), float("inf"), -1.0])
    depth[1, 5, 7] = 0.05                                                 # valid, below d_min -> 0.2
    out = norm(rgb8, depth)
    assert out.image.shape == (b, 4, h, w) and out.depth_m.shape == (b, h, w) and out.valid.dtype == torch.bool
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64).view(1, 3, 1, 1)
    assert _close(out.image[:, :3], (rgb8.double() / 255.0 - mean) / std)
    out_f = norm(rgb8.float() / 255.0, depth.unsqueeze(1))                 # float RGB, [B, 1, H, W] depth
    assert torch.allclose(out_f.image, out.image, rtol=_RTOL, atol=_ATOL)
    valid = out.valid
    assert torch.equal(valid, torch.isfinite(depth) & (depth > 0)) and valid[0, 0, :4].tolist() == [False] * 4
    assert float(out.image[0, 3, 0, :4].abs().sum()) == 0.0 and float(out.depth_m[0, 0, :4].abs().sum()) == 0.0
    d_ref = depth.double().clamp(0.2, 8.0)
    assert _close(out.depth_m[valid], d_ref[valid]) and _close(out.image[:, 3][valid], d_ref[valid] / 8.0)
    assert abs(float(out.depth_m[1, 5, 7]) - 0.2) < 1e-6 and float(out.image[:, 3].max()) <= 1.0
    with torch.no_grad():
        assert enc(out.image).shape == (b, d)                             # normaliser output feeds E_psi
    for bad in ((rgb8[:, :2], depth), (rgb8, depth[:, :-1]), (rgb8, depth.unsqueeze(1).expand(b, 2, h, w))):
        _expect_value_error(lambda a=bad: norm(*a))
    for kw in (dict(depth_range=(0.0, 8.0)), dict(depth_range=(5.0, 1.0)), dict(rgb_std=(0.2, 0.0, 0.2))):
        _expect_value_error(lambda k=kw: RGBDNormalizer(**k))

    # ---- Momentum target E_psi_bar (Eq. 8): frozen copy, EMA update, stop-gradient.
    target_full = MomentumEncoder(enc)
    assert count_parameters(target_full) == 0
    assert count_parameters(target_full, trainable_only=False) == AERO_ENCODER_PARAMS
    mu = 0.9
    target = MomentumEncoder(small, momentum=mu)
    for p_bar, p in zip(target.encoder.parameters(), small.parameters(), strict=True):
        assert torch.equal(p_bar, p) and p_bar.data_ptr() != p.data_ptr()  # copy, not alias
    psi_bar_ref = [p.detach().double().clone() for p in small.parameters()]
    for _ in range(5):
        with torch.no_grad():  # simulated optimiser step on the online encoder
            for p in small.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=gen))
        psi_before = [p.detach().clone() for p in small.parameters()]
        target.update(small)
        for i, p in enumerate(small.parameters()):
            psi_bar_ref[i] = mu * psi_bar_ref[i] + (1.0 - mu) * p.detach().double()
        for p_bar, p_ref, p, p_old in zip(
            target.encoder.parameters(), psi_bar_ref, small.parameters(), psi_before, strict=True
        ):
            assert _close(p_bar, p_ref)
            assert torch.equal(p, p_old)  # update() never writes to the online encoder
    assert _close(target(x_small), _ref_vit_forward(target.encoder, x_small))
    x_g = x_small.clone().requires_grad_(True)
    z_tgt = target(x_g)
    assert not z_tgt.requires_grad and z_tgt.grad_fn is None
    small.zero_grad(set_to_none=True)
    ((small(x_g) - z_tgt) ** 2).sum().backward()
    assert all(p.grad is not None for p in small.parameters())
    assert all(p.grad is None for p in target.parameters())
    _expect_value_error(lambda: MomentumEncoder(small, momentum=1.0))
    _expect_value_error(lambda: MomentumEncoder(small, momentum=-0.1))

    # ---- c_text (Sec. 3.2): frozen tower (offline stand-in with the CLIP interface, d_c = 512)
    #      and the learned null embedding, the wrapper's only trainable parameter.
    text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=512), tokenizer=_toy_tokenizer)
    assert text.embed_dim == 512 and count_parameters(text) == 512
    for mode in (None, True):
        text.train() if mode is None else text.train(mode)
        assert not text.text_model.training
        assert not any(mod.training for mod in text.text_model.modules())
    assert not any(p.requires_grad for p in text.text_model.parameters())
    texts = ["fly to the red gate", "inspect the second floor", "return home"]
    c_text = text(texts)
    assert c_text.shape == (3, 512) and not c_text.requires_grad and c_text.grad_fn is None
    tokens = _toy_tokenizer(texts, padding=True, truncation=True, max_length=text.max_length, return_tensors="pt")
    with torch.no_grad():
        direct = text.text_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).text_embeds
    assert torch.equal(c_text, direct)
    with torch.no_grad():
        text.null_embedding.copy_(torch.randn(512, generator=gen))
    c_null = text(None)
    assert c_null.shape == (1, 512) and torch.equal(c_null[0], text.null_embedding.detach())
    assert torch.equal(text("  ")[0], text.null_embedding.detach())
    mixed = text(["fly to the red gate", None, "", "return home"])
    assert mixed.shape == (4, 512)
    assert torch.allclose(mixed[0], c_text[0], rtol=_RTOL, atol=_ATOL)   # B = 2 vs B = 3 BLAS rounding
    assert torch.allclose(mixed[3], c_text[2], rtol=_RTOL, atol=_ATOL)
    assert torch.equal(mixed[1], text.null_embedding.detach()) and torch.equal(mixed[2], mixed[1])
    mixed.sum().backward()
    assert torch.equal(text.null_embedding.grad, torch.full((512,), 2.0))  # two null rows
    assert all(p.grad is None for p in text.text_model.parameters())
    _expect_value_error(lambda: text([]))
    print("perception.py self-test passed")


if __name__ == "__main__":
    _self_test()
