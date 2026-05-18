"""Stage 4: optional joint generation + multi-positive contrastive training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from atlas_free_multipositive.training.checkpointing import CheckpointManager
from atlas_free_multipositive.training.collators import MultiPositiveCollator
from atlas_free_multipositive.training.datasets import UnifiedMapTextDataset
from atlas_free_multipositive.training.generation_losses import GenerationLossConfig, combined_generation_loss
from atlas_free_multipositive.training.losses import multi_positive_infonce
from atlas_free_multipositive.training.model_wrappers import (
    build_cnn_autoencoder,
    build_text_projection,
    load_autoencoder_checkpoint,
    load_text_projection_checkpoint,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _target_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cfg.get("target_shape", [36, 45, 38]))


def _text_cache(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload.items()}


def _lookup(cache, texts):
    return torch.stack([cache[t] for t in texts])


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    target_shape = _target_shape(cfg)
    model_cfg = cfg.get("model", {})
    autoencoder = build_cnn_autoencoder(
        target_shape,
        latent_dim=int(model_cfg.get("latent_dim", 384)),
        base_channels=int(model_cfg.get("base_channels", 8)),
        num_blocks=int(model_cfg.get("num_blocks", 2)),
        encoder_arch=str(model_cfg.get("encoder_arch", "plain")),
        blocks_per_stage=int(model_cfg.get("blocks_per_stage", 2)),
        use_dilation=bool(model_cfg.get("use_dilation", False)),
        multi_scale=bool(model_cfg.get("multi_scale", False)),
        global_context=str(model_cfg.get("global_context", "none")),
    ).to(device)
    load_autoencoder_checkpoint(autoencoder, cfg["autoencoder_checkpoint"])
    text_projector = build_text_projection(cfg.get("text_projection_init", "random"), device=device)
    if cfg.get("text_projection_checkpoint"):
        load_text_projection_checkpoint(text_projector, cfg["text_projection_checkpoint"])
    if cfg.get("freeze_decoder", True):
        for p in autoencoder.decoder.parameters():
            p.requires_grad_(False)
    if cfg.get("freeze_encoder", False):
        for p in autoencoder.encoder.parameters():
            p.requires_grad_(False)
    ds = UnifiedMapTextDataset(cfg["train_jsonl"])
    collator = MultiPositiveCollator(positives_per_map=int(cfg.get("positive_texts_per_map", 2)), target_shape=target_shape, seed=int(cfg.get("seed", 42)))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 4)), shuffle=True, num_workers=int(cfg.get("num_workers", 0)), collate_fn=collator)
    cache = _text_cache(cfg["text_embedding_cache"])
    params = [p for p in list(autoencoder.encoder.parameters()) + list(autoencoder.decoder.parameters()) + list(text_projector.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_weights = cfg.get("loss", {})
    gen_cfg = GenerationLossConfig(lambda_latent=float(loss_weights.get("lambda_latent", 0.25)))
    ckpt = CheckpointManager(cfg.get("checkpoint_dir", "atlas_free_multipositive/outputs/runs/joint_generation_contrastive/checkpoints"))
    history = []
    for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
        autoencoder.train()
        if cfg.get("freeze_decoder", True):
            autoencoder.decoder.eval()
        text_projector.train()
        losses = []
        for step, batch in enumerate(loader):
            if cfg.get("max_train_batches") is not None and step >= int(cfg["max_train_batches"]):
                break
            x = batch["volume"].to(device)
            raw_text = _lookup(cache, batch["texts"]).to(device)
            brain_z = autoencoder.encoder(x)
            text_z = text_projector(raw_text)
            contrastive = multi_positive_infonce(brain_z, text_z, batch["pos_mask"].to(device), batch["pos_weights"].to(device), temperature=float(cfg.get("temperature", 0.07)))
            owners = batch["pos_mask"].float().argmax(dim=0).to(device)
            target = x[owners]
            pred = autoencoder.decoder(text_z)
            gen_loss, parts = combined_generation_loss(pred, target, brain_z=brain_z[owners], text_z=text_z, config=gen_cfg)
            loss = float(loss_weights.get("lambda_contrastive", 1.0)) * contrastive + float(loss_weights.get("lambda_generation", 0.25)) * gen_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses))}
        history.append(row)
        ckpt.save_last({"autoencoder": autoencoder.state_dict(), "text_projector": text_projector.state_dict(), "config": cfg, "history": history})
        print(row)
    return {"history": history, "checkpoint_dir": str(ckpt.out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="atlas_free_multipositive/configs/joint_generation_contrastive_config.yaml")
    args = p.parse_args()
    train_from_config(load_yaml(args.config))


if __name__ == "__main__":
    main()
