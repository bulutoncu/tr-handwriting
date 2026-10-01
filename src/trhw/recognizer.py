"""Base-shape line recognizer built on TrOCR (vision encoder + text decoder).

The recognizer is trained to output the BASE text ("Cocuk okula gitti"), so it
never has to see a Turkish mark. Marks are restored afterwards by the
restorers in `restore.py`.

A plain PyTorch training loop is used on purpose instead of a high-level
Trainer: every step (batching, loss, gradient accumulation, scheduling,
evaluation, checkpointing) is visible and easy to change.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .metrics import cer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def read_labels(data_dir: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(Path(data_dir) / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_image(data_dir: str | Path, row: dict) -> Image.Image:
    return Image.open(Path(data_dir) / row["image"]).convert("RGB")


class LineDataset(Dataset):
    """Line images + base-shape labels from a folder made by generate_synth.py
    (or by the real-data import tool later, same format)."""

    def __init__(self, data_dir, processor, limit=None, field="base", max_target_len=96):
        self.data_dir = Path(data_dir)
        self.rows = read_labels(data_dir, limit)
        self.processor = processor
        self.field = field
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        pixels = self.processor.image_processor(load_image(self.data_dir, row),
                                                return_tensors="pt").pixel_values[0]
        ids = self.processor.tokenizer(row[self.field], max_length=self.max_target_len,
                                       truncation=True).input_ids
        return {"pixel_values": pixels, "labels": torch.tensor(ids, dtype=torch.long)}


def collate(batch: list[dict]) -> dict:
    pixels = torch.stack([b["pixel_values"] for b in batch])
    longest = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)  # -100 = ignored by the loss
    for k, b in enumerate(batch):
        labels[k, :len(b["labels"])] = b["labels"]
    return {"pixel_values": pixels, "labels": labels}


def load_model(name_or_path: str, device: torch.device | None = None):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(name_or_path)
    prepare_model(model, processor.tokenizer)
    return model.to(device or pick_device()), processor


def prepare_model(model, tokenizer) -> None:
    """Make sure the special-token ids needed for training and generation are set."""
    cfg = model.config
    # official TrOCR checkpoints already carry these; fill them in only if missing
    if getattr(cfg, "decoder_start_token_id", None) is None:
        cfg.decoder_start_token_id = getattr(cfg.decoder, "decoder_start_token_id", None) \
            or tokenizer.eos_token_id
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tokenizer.pad_token_id
    if getattr(cfg, "eos_token_id", None) is None:
        cfg.eos_token_id = tokenizer.eos_token_id
    gen = model.generation_config
    gen.decoder_start_token_id = cfg.decoder_start_token_id
    gen.pad_token_id = cfg.pad_token_id
    gen.eos_token_id = cfg.eos_token_id


@torch.no_grad()
def predict(model, processor, images: list[Image.Image], batch_size: int = 16,
            num_beams: int = 1, max_new_tokens: int = 96) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    out: list[str] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        pixels = processor.image_processor(chunk, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pixels, num_beams=num_beams, max_new_tokens=max_new_tokens)
        out.extend(t.strip() for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
    return out


def evaluate_cer(model, processor, data_dir, n: int = 300, batch_size: int = 16) -> float:
    rows = read_labels(data_dir, n)
    preds = predict(model, processor, [load_image(data_dir, r) for r in rows], batch_size)
    return cer([r["base"] for r in rows], preds)


@dataclass
class TrainConfig:
    epochs: float = 1.0
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_share: float = 0.05
    eval_every: int = 500           # optimizer steps
    eval_n: int = 300               # validation lines per evaluation
    log_every: int = 50
    num_workers: int = 2
    max_grad_norm: float = 1.0
    seed: int = 0


def train(model, processor, train_ds, val_dir, out_dir, cfg: TrainConfig) -> dict:
    """Fine-tune and keep the checkpoint with the best validation CER."""
    torch.manual_seed(cfg.seed)
    device = next(model.parameters()).device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    total_steps = max(1, math.ceil(len(loader) * cfg.epochs / cfg.grad_accum))
    warmup = max(1, int(total_steps * cfg.warmup_share))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # linear warmup, then linear decay to zero
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / (total_steps - warmup + 1e-9))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_file = open(out_dir / "history.csv", "w", newline="")
    history = csv.writer(history_file)
    history.writerow(["step", "train_loss", "val_cer", "lr", "seconds"])

    def evaluate_and_maybe_save(step, running_loss):
        nonlocal best
        val = evaluate_cer(model, processor, val_dir, cfg.eval_n, cfg.batch_size)
        model.train()
        history.writerow([step, f"{running_loss:.4f}", f"{val:.4f}",
                          f"{schedule.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
        history_file.flush()
        marker = ""
        if val < best:
            best = val
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            marker = "  <- best, saved"
        print(f"step {step}/{total_steps}  val CER {val:.2%}{marker}")

    best = evaluate_cer(model, processor, val_dir, cfg.eval_n, cfg.batch_size)
    model.save_pretrained(out_dir / "best")  # so a checkpoint exists even if nothing improves
    processor.save_pretrained(out_dir / "best")
    print(f"before training: val CER {best:.2%}")
    t0 = time.time()
    step, micro, window_loss, window_steps, running = 0, 0, 0.0, 0, float("nan")
    model.train()
    while step < total_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % cfg.grad_accum:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            window_loss += loss.item() * cfg.grad_accum
            window_steps += 1
            if step % cfg.log_every == 0:
                running = window_loss / window_steps  # mean loss since the last log line
                window_loss, window_steps = 0.0, 0
                rate = step / (time.time() - t0)
                eta = (total_steps - step) / max(rate, 1e-9) / 60
                print(f"step {step}/{total_steps}  loss {running:.3f}  {rate:.2f} steps/s  ~{eta:.0f} min left")
            if step % cfg.eval_every == 0 or step == total_steps:
                evaluate_and_maybe_save(step, running)
            if step >= total_steps:
                break
    history_file.close()
    return {"best_val_cer": best, "steps": step, "minutes": (time.time() - t0) / 60}
