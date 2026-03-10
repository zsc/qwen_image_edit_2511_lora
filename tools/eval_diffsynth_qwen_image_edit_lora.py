#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch

from fid_torchvision import compute_fid


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_split_ids(dataset_dir: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    split_path = dataset_dir / "split.json"
    if not split_path.exists():
        return None
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    return set(obj.get(split, []))


def _crop_resize_like_training(image: Image.Image, *, max_pixels: int) -> Image.Image:
    # Mirror DiffSynth ImageCropAndResize (dynamic size, division_factor=16).
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"bad image size: {image.size}")

    if w * h > max_pixels:
        scale = (w * h / max_pixels) ** 0.5
        h = int(h / scale)
        w = int(w / scale)

    # floor to division factor (16)
    h = (h // 16) * 16
    w = (w // 16) * 16
    h = max(h, 16)
    w = max(w, 16)

    # Resize then center crop to (h, w)
    ow, oh = image.size
    scale = max(w / ow, h / oh)
    rh = int(round(oh * scale))
    rw = int(round(ow * scale))
    resized = image.resize((rw, rh), resample=Image.BILINEAR)
    left = max(0, (rw - w) // 2)
    top = max(0, (rh - h) // 2)
    return resized.crop((left, top, left + w, top + h))


def _crop_resize_to(image: Image.Image, *, target_w: int, target_h: int) -> Image.Image:
    ow, oh = image.size
    if ow <= 0 or oh <= 0:
        raise ValueError(f"bad image size: {image.size}")
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"bad target size: {(target_w, target_h)}")
    scale = max(target_w / ow, target_h / oh)
    rw = int(round(ow * scale))
    rh = int(round(oh * scale))
    resized = image.resize((rw, rh), resample=Image.BILINEAR)
    left = max(0, (rw - target_w) // 2)
    top = max(0, (rh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def _detect_diffsynth_repo(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for env_name in ("DIFFSYNTH_DIR",):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            Path("/workspace/zhousc6@xiaopeng.com/DiffSynth-Studio"),
            Path("/root/autodl-tmp/DiffSynth-Studio"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        "DiffSynth-Studio not found. Pass --diffsynth_repo or set DIFFSYNTH_DIR."
    )


def _detect_model_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for env_name in ("QWEN_IMAGE_EDIT_MODEL_DIR",):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            Path("/root/hf_models/Qwen/Qwen-Image-Edit-2511"),
            Path("/workspace/zhousc6@xiaopeng.com/hf_models/Qwen/Qwen-Image-Edit-2511"),
        ]
    )
    for path in candidates:
        if (path / "transformer").exists() and (path / "text_encoder").exists() and (path / "vae").exists():
            return path
    raise SystemExit(
        "Qwen-Image-Edit-2511 model dir not found. Pass --model_dir or set QWEN_IMAGE_EDIT_MODEL_DIR."
    )

def _to_float_rgb(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a - b)))


def _dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _maybe_init_dist() -> tuple[int, int, int]:
    world_size, rank, local_rank = _dist_env()
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    return world_size, rank, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _destroy_dist(world_size: int) -> None:
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a DiffSynth Qwen-Image-Edit LoRA checkpoint on a dataset.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Prepared dataset dir (qwen_edit_demo prepare output).")
    parser.add_argument("--lora", type=Path, default=None, help="Optional LoRA checkpoint (.safetensors), e.g. step-2000.safetensors")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for eval artifacts.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "all"), help="Which split to sample from.")
    parser.add_argument("--num_samples", type=int, default=8, help="How many samples to run.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling + inference seed.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Diffusion steps for inference.")
    parser.add_argument("--max_pixels", type=int, default=1048576, help="Max pixels (match training).")
    parser.add_argument(
        "--low_vram",
        type=int,
        default=1,
        choices=(0, 1),
        help="Use DiffSynth VRAM management (recommended if training is running on the same GPU).",
    )
    parser.add_argument(
        "--vram_limit_gb",
        type=float,
        default=0.0,
        help="If >0, limit DiffSynth VRAM preloading/caching to this many GiB (helps avoid OOM when training holds most VRAM).",
    )
    parser.add_argument(
        "--computation_dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp8"),
        help="DType for VRAM-managed model compute. Use fp8 to reduce eval VRAM.",
    )
    parser.add_argument("--compute_fid", type=int, default=1, choices=(0, 1), help="Compute FID on generated preds.")
    parser.add_argument("--fid_device", type=str, default="cuda", help="Device for Inception FID feature extraction.")
    parser.add_argument("--fid_batch_size", type=int, default=32, help="Batch size for Inception feature extraction.")
    parser.add_argument("--diffsynth_repo", type=Path, default=None, help="DiffSynth-Studio repo path.")
    parser.add_argument("--model_dir", type=Path, default=None, help="Local model dir containing transformer/text_encoder/vae/tokenizer/processor.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2511", help="Model id used by DiffSynth path resolution.")
    args = parser.parse_args()
    world_size, rank, local_rank = _maybe_init_dist()

    ds_root = _detect_diffsynth_repo(args.diffsynth_repo)
    model_dir = _detect_model_dir(args.model_dir)
    sys.path.insert(0, str(ds_root))

    # Ensure offline/local model loading.
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(model_dir.parent.parent))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig  # noqa: E402

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.dataset_dir / "metadata.diffsynth.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"missing: {meta_path}")

    rows = _load_jsonl(meta_path)
    split_ids = _load_split_ids(args.dataset_dir, args.split)
    if split_ids is not None:
        rows = [r for r in rows if Path(r["image"]).stem in split_ids]
    if not rows:
        raise SystemExit("no samples after split filtering")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.num_samples]
    rows_with_index = list(enumerate(rows))
    shard_rows = rows_with_index[rank::world_size]
    shard_json = args.out_dir / f"results.rank{rank}.json"
    if not shard_rows:
        shard_json.write_text(json.dumps({"rank": rank, "world_size": world_size, "results": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"rank={rank} has no rows to process", flush=True)
        print(f"rank={rank} wrote: {shard_json}", flush=True)
        _barrier(world_size)
        if rank != 0:
            _destroy_dist(world_size)
            return 0

    float8 = getattr(torch, "float8_e4m3fn", torch.float16)
    if args.computation_dtype == "bf16":
        computation_dtype = torch.bfloat16
    elif args.computation_dtype == "fp16":
        computation_dtype = torch.float16
    else:
        computation_dtype = getattr(torch, "float8_e4m3fn", None)
        if computation_dtype is None:
            raise SystemExit("torch has no float8 support; cannot use --computation_dtype fp8")
    vram_config = None
    device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    if args.low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": float8,
            "onload_device": "cpu",
            "preparing_dtype": float8,
            "preparing_device": device,
            "computation_dtype": computation_dtype,
            "computation_device": device,
        }
    vram_limit = float(args.vram_limit_gb) if args.low_vram and args.vram_limit_gb and args.vram_limit_gb > 0 else None

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **(vram_config or {}),
            ),
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="text_encoder/model*.safetensors",
                **(vram_config or {}),
            ),
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **(vram_config or {}),
            ),
        ],
        tokenizer_config=ModelConfig(str(model_dir / "tokenizer")),
        processor_config=ModelConfig(str(model_dir / "processor")),
        vram_limit=vram_limit,
    )
    if args.lora is not None:
        pipe.load_lora(pipe.dit, str(args.lora))

    results: list[dict[str, Any]] = []
    for order_index, r in shard_rows:
        sample_id = Path(r["image"]).stem
        tgt_path = args.dataset_dir / r["image"]
        ctrl_path = args.dataset_dir / r["edit_image"]

        tgt = Image.open(tgt_path).convert("RGBA")
        ctrl = Image.open(ctrl_path).convert("RGB")

        tgt_p = _crop_resize_like_training(tgt, max_pixels=args.max_pixels)
        ctrl_p = _crop_resize_like_training(ctrl, max_pixels=args.max_pixels)
        w, h = tgt_p.size
        ctrl_to_tgt = _crop_resize_to(ctrl, target_w=w, target_h=h)

        pred = pipe(
            prompt=r.get("prompt", "mel-spectrum"),
            negative_prompt="",
            cfg_scale=1,
            edit_image=[ctrl_p],
            seed=args.seed + order_index,
            num_inference_steps=args.num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            zero_cond_t=True,
            progress_bar_cmd=(lambda x: x),
        )

        # Save artifacts
        sample_dir = args.out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        ctrl_p.save(sample_dir / "control.png")
        ctrl_to_tgt.save(sample_dir / "control_to_target.png")
        tgt_p.save(sample_dir / "target.png")
        pred.save(sample_dir / "pred.png")

        # Quick pixel-space metrics
        l1_ctrl = _l1(_to_float_rgb(ctrl_to_tgt), _to_float_rgb(tgt_p))
        l1_pred = _l1(_to_float_rgb(pred), _to_float_rgb(tgt_p))
        results.append(
            {
                "id": sample_id,
                "width": int(w),
                "height": int(h),
                "control_w": int(ctrl_p.size[0]),
                "control_h": int(ctrl_p.size[1]),
                "l1_control_to_target": l1_ctrl,
                "l1_pred_to_target": l1_pred,
                "improve_l1": float(l1_ctrl - l1_pred),
                "sample_index": int(order_index),
                "rank": int(rank),
            }
        )

    shard_json.write_text(json.dumps({"rank": rank, "world_size": world_size, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rank={rank} wrote: {shard_json}", flush=True)
    _barrier(world_size)

    if rank != 0:
        _destroy_dist(world_size)
        return 0

    merged_results: list[dict[str, Any]] = []
    for shard_rank in range(world_size):
        shard_path = args.out_dir / f"results.rank{shard_rank}.json"
        if not shard_path.exists():
            raise SystemExit(f"missing shard output: {shard_path}")
        shard_obj = json.loads(shard_path.read_text(encoding="utf-8"))
        merged_results.extend(shard_obj.get("results", []))
    merged_results.sort(key=lambda item: item.get("sample_index", 0))

    metrics: dict[str, Any] = {
        "results": merged_results,
        "lora_path": str(args.lora) if args.lora is not None else None,
        "model_dir": str(model_dir),
        "model_id": args.model_id,
        "split": args.split,
        "num_inference_steps": args.num_inference_steps,
        "world_size": world_size,
    }

    if args.compute_fid:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        real_paths = [args.out_dir / r["id"] / "target.png" for r in merged_results]
        fake_paths = [args.out_dir / r["id"] / "pred.png" for r in merged_results]
        fid_res = compute_fid(real_paths, fake_paths, device=args.fid_device, batch_size=args.fid_batch_size)
        metrics["fid"] = {
            "value": fid_res.fid,
            "n_real": fid_res.n_real,
            "n_fake": fid_res.n_fake,
            "feature_dim": fid_res.feature_dim,
            "impl": fid_res.impl,
        }

    out_json = args.out_dir / "metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mean_improve = float(np.mean([r["improve_l1"] for r in merged_results])) if merged_results else 0.0
    print(f"wrote: {out_json}")
    print(f"mean improve_l1: {mean_improve:.6f} (positive is better)")
    if "fid" in metrics:
        print(f"fid: {metrics['fid']['value']:.6f} ({metrics['fid']['impl']})")
    _destroy_dist(world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
