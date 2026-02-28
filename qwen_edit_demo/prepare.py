"""Prepare training dataset from scanned samples."""

import json
from pathlib import Path
from typing import List, Optional

from .utils import (
    ensure_dir,
    copy_file,
    get_now_iso,
    validate_copy_mode,
    validate_val_ratio,
    split_train_val,
)


def prepare_dataset(
    samples: List[dict],
    out_dir: Path,
    val_ratio: float = 0.0,
    seed: int = 42,
    copy_mode: str = "symlink",
    src_dir: Optional[Path] = None,
    tgt_dir: Optional[Path] = None,
) -> dict:
    """
    Prepare training dataset from scanned samples.
    
    Args:
        samples: List of sample dicts from scan.scan_directory()
        out_dir: Output directory for prepared dataset
        val_ratio: Validation set ratio (0-0.5)
        seed: Random seed for train/val split
        copy_mode: One of "copy", "hardlink", "symlink"
        src_dir: Original source directory (for manifest)
        tgt_dir: Original target directory (for manifest)
    
    Returns:
        Dict with manifest info
    """
    copy_mode = validate_copy_mode(copy_mode)
    val_ratio = validate_val_ratio(val_ratio)
    
    out_dir = Path(out_dir)
    
    # Create output directories
    control_dir = ensure_dir(out_dir / "images" / "control")
    target_dir = ensure_dir(out_dir / "images" / "target")
    prompts_dir = ensure_dir(out_dir / "prompts")
    
    # Split train/val
    all_ids = [s["id"] for s in samples]
    train_ids, val_ids = split_train_val(all_ids, val_ratio, seed)
    
    # Process samples
    metadata = []
    
    for sample in samples:
        sample_id = sample["id"]
        src_img = sample["src_image"]
        tgt_img = sample["tgt_image"]
        prompt = sample["prompt"]
        
        # Preserve extensions for control/target independently.
        control_ext = src_img.suffix
        target_ext = tgt_img.suffix
        
        # Define output paths
        control_path = control_dir / f"{sample_id}{control_ext}"
        target_path = target_dir / f"{sample_id}{target_ext}"
        prompt_path = prompts_dir / f"{sample_id}.txt"
        
        # Copy/link files
        copy_file(src_img, control_path, copy_mode)
        copy_file(tgt_img, target_path, copy_mode)
        
        # Write prompt file
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        # Create metadata entry (relative paths)
        meta_entry = {
            "id": sample_id,
            "control_image": str(Path("images/control") / f"{sample_id}{control_ext}"),
            "image": str(Path("images/target") / f"{sample_id}{target_ext}"),
            "prompt": prompt,
        }
        metadata.append(meta_entry)
    
    # Write metadata.jsonl
    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Write manifest.json
    manifest = {
        "num_total": len(samples),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "exts": list(
            sorted(
                {
                    s["src_image"].suffix.lstrip(".").lower()
                    for s in samples
                }
                | {
                    s["tgt_image"].suffix.lstrip(".").lower()
                    for s in samples
                }
            )
        ),
        "created_at": get_now_iso(),
        "src_dir": str(src_dir) if src_dir else None,
        "tgt_dir": str(tgt_dir) if tgt_dir else None,
        "copy_mode": copy_mode,
        "seed": seed,
        "val_ratio": val_ratio,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Write split.json if val_ratio > 0
    if val_ratio > 0:
        split_data = {
            "train": train_ids,
            "val": val_ids,
        }
        split_path = out_dir / "split.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    return manifest


def validate_prepared_dataset(dataset_dir: Path) -> dict:
    """
    Validate a prepared dataset directory.
    
    Returns:
        Dict with validation results
    """
    dataset_dir = Path(dataset_dir)
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    
    # Check required files
    metadata_path = dataset_dir / "metadata.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    control_dir = dataset_dir / "images" / "control"
    target_dir = dataset_dir / "images" / "target"
    prompts_dir = dataset_dir / "prompts"
    
    if not metadata_path.exists():
        results["valid"] = False
        results["errors"].append("metadata.jsonl not found")
    
    if not manifest_path.exists():
        results["valid"] = False
        results["errors"].append("manifest.json not found")
    
    # Validate metadata.jsonl
    if metadata_path.exists():
        entries = []
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                        
                        # Check required fields
                        for field in ["id", "control_image", "image", "prompt"]:
                            if field not in entry:
                                results["valid"] = False
                                results["errors"].append(
                                    f"Line {line_num}: missing field '{field}'"
                                )
                    except json.JSONDecodeError as e:
                        results["valid"] = False
                        results["errors"].append(f"Line {line_num}: invalid JSON - {e}")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Error reading metadata.jsonl: {e}")
        
        # Check file existence
        for entry in entries:
            entry_id = entry.get("id", "unknown")
            
            control_rel = entry.get("control_image")
            if control_rel:
                control_path = dataset_dir / control_rel
                if not control_path.exists():
                    results["warnings"].append(
                        f"ID {entry_id}: control_image not found: {control_rel}"
                    )
            
            target_rel = entry.get("image")
            if target_rel:
                target_path = dataset_dir / target_rel
                if not target_path.exists():
                    results["warnings"].append(
                        f"ID {entry_id}: image not found: {target_rel}"
                    )
    
    return results
