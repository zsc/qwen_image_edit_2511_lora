"""Utility functions for qwen_edit_demo."""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Tuple, Optional


DEFAULT_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}


def normalize_exts(exts_str: Optional[str]) -> Set[str]:
    """Parse comma-separated extensions string into a set of lowercase extensions with dots."""
    if not exts_str:
        return DEFAULT_EXTS.copy()
    exts = set()
    for ext in exts_str.split(","):
        ext = ext.strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        exts.add(ext)
    return exts if exts else DEFAULT_EXTS.copy()


def get_file_stem(filepath: Path) -> str:
    """Get the stem (filename without extension) of a file."""
    return filepath.stem


def is_image_file(filepath: Path, allowed_exts: Set[str]) -> bool:
    """Check if a file is an image with allowed extension."""
    return filepath.suffix.lower() in allowed_exts


def read_prompt_txt(txt_path: Path) -> str:
    """
    Read prompt text from file.
    
    Tries utf-8 first, then utf-8-sig (for BOM).
    Returns stripped text, or "-" if empty.
    
    Raises:
        UnicodeDecodeError: If file cannot be decoded as UTF-8 / UTF-8-SIG.
    """
    errors = []
    
    # Try utf-8
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text if text else "-"
    except UnicodeDecodeError as e:
        errors.append(f"utf-8: {e}")
    
    # Try utf-8-sig (for BOM)
    try:
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
            return text if text else "-"
    except UnicodeDecodeError as e:
        errors.append(f"utf-8-sig: {e}")
    
    # If both fail, raise error
    raise UnicodeDecodeError(
        "utf-8",
        str(txt_path).encode(),
        0,
        0,
        f"Failed to decode {txt_path} with encodings: {errors}"
    )


def compute_file_hash(filepath: Path, algorithm: str = "md5") -> Optional[str]:
    """Compute hash of a file. Returns None if file doesn't exist."""
    if not filepath.exists():
        return None
    
    hasher = hashlib.new(algorithm)
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path, mode: str = "symlink") -> None:
    """
    Copy or link a file from src to dst.
    
    Args:
        src: Source file path
        dst: Destination file path
        mode: One of "copy", "hardlink", "symlink"
    """
    ensure_dir(dst.parent)
    
    # Remove existing file/link if exists
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    
    if mode == "copy":
        import shutil
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        # Use absolute path for symlink to avoid issues with relative paths
        src_abs = src.resolve()
        os.symlink(src_abs, dst)
    else:
        raise ValueError(f"Unknown copy_mode: {mode}")


def get_now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_copy_mode(mode: str) -> str:
    """Validate and normalize copy mode."""
    mode = mode.lower().strip()
    valid_modes = {"copy", "hardlink", "symlink"}
    if mode not in valid_modes:
        raise ValueError(f"copy_mode must be one of {valid_modes}, got: {mode}")
    return mode


def validate_val_ratio(ratio: float) -> float:
    """Validate validation ratio."""
    if not 0 <= ratio <= 0.5:
        raise ValueError(f"val_ratio must be between 0 and 0.5, got: {ratio}")
    return ratio


def split_train_val(ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Split list of IDs into train and validation sets.
    
    Returns:
        Tuple of (train_ids, val_ids)
    """
    import random
    
    if val_ratio <= 0:
        return ids.copy(), []
    
    if val_ratio >= 1:
        return [], ids.copy()
    
    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_ids = shuffled[:n_val]
    train_ids = shuffled[n_val:]
    
    return train_ids, val_ids
