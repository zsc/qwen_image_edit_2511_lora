"""Scan and validate image pairs for training data preparation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .utils import (
    normalize_exts,
    is_image_file,
    read_prompt_txt,
    get_file_stem,
)


@dataclass
class ScanResult:
    """Result of scanning directories."""
    samples: List[dict]  # List of valid samples
    missing_src: List[str]  # IDs missing in src_dir
    missing_tgt_image: List[str]  # IDs missing target image
    missing_tgt_txt: List[str]  # IDs missing txt file
    duplicate_ids: Dict[str, List[Path]]  # IDs with multiple images
    warnings: List[str]  # Non-fatal warnings
    errors: List[str]  # Other errors
    
    @property
    def num_total_ids(self) -> int:
        """Total number of unique IDs found."""
        all_ids = set()
        for s in self.samples:
            all_ids.add(s["id"])
        all_ids.update(self.missing_src)
        all_ids.update(self.missing_tgt_image)
        all_ids.update(self.missing_tgt_txt)
        all_ids.update(self.duplicate_ids.keys())
        return len(all_ids)
    
    @property
    def num_valid_samples(self) -> int:
        """Number of valid samples."""
        return len(self.samples)
    
    @property
    def has_issues(self) -> bool:
        """True if there are any issues."""
        return bool(
            self.missing_src or 
            self.missing_tgt_image or 
            self.missing_tgt_txt or 
            self.duplicate_ids or 
            self.errors
        )


def scan_directory(
    src_dir: Path,
    tgt_dir: Path,
    exts: Optional[str] = None,
    recursive: bool = False,
) -> ScanResult:
    """
    Scan source and target directories for valid image pairs.
    
    Args:
        src_dir: Directory containing source images (A/)
        tgt_dir: Directory containing target images and txt files (B/)
        exts: Comma-separated list of allowed extensions (default: png,jpg,jpeg,webp)
        recursive: Whether to scan subdirectories
    
    Returns:
        ScanResult containing samples and issues
    """
    allowed_exts = normalize_exts(exts)
    
    # Scan for images
    if recursive:
        src_images = list(src_dir.rglob("*"))
        tgt_files = list(tgt_dir.rglob("*"))
    else:
        src_images = list(src_dir.iterdir()) if src_dir.exists() else []
        tgt_files = list(tgt_dir.iterdir()) if tgt_dir.exists() else []
    
    # Filter to only files
    src_images = [f for f in src_images if f.is_file()]
    tgt_files = [f for f in tgt_files if f.is_file()]
    
    # Filter source images by extension
    src_images = [f for f in src_images if is_image_file(f, allowed_exts)]
    
    # Separate target images and txt files
    tgt_images = [f for f in tgt_files if is_image_file(f, allowed_exts)]
    tgt_txts = [f for f in tgt_files if f.suffix.lower() == ".txt"]
    
    # Build maps by stem
    src_map: Dict[str, Path] = {}
    src_duplicates: Dict[str, List[Path]] = {}
    
    for img in src_images:
        stem = get_file_stem(img)
        if stem in src_map:
            if stem not in src_duplicates:
                src_duplicates[stem] = [src_map[stem]]
            src_duplicates[stem].append(img)
        else:
            src_map[stem] = img
    
    tgt_img_map: Dict[str, Path] = {}
    tgt_duplicates: Dict[str, List[Path]] = {}
    
    for img in tgt_images:
        stem = get_file_stem(img)
        if stem in tgt_img_map:
            if stem not in tgt_duplicates:
                tgt_duplicates[stem] = [tgt_img_map[stem]]
            tgt_duplicates[stem].append(img)
        else:
            tgt_img_map[stem] = img
    
    # Build target txt map deterministically and track duplicates (non-fatal).
    tgt_txt_map: Dict[str, Path] = {}
    tgt_txt_duplicates: Dict[str, List[Path]] = {}
    txt_groups: Dict[str, List[Path]] = {}
    for txt in tgt_txts:
        txt_groups.setdefault(get_file_stem(txt), []).append(txt)
    for stem, paths in txt_groups.items():
        sorted_paths = sorted(paths, key=lambda p: str(p))
        tgt_txt_map[stem] = sorted_paths[0]
        if len(sorted_paths) > 1:
            tgt_txt_duplicates[stem] = sorted_paths
    
    # Combine all IDs
    all_ids = set(src_map.keys()) | set(tgt_img_map.keys()) | set(tgt_txt_map.keys())
    
    # Combine duplicates from both dirs without overwriting.
    all_duplicates: Dict[str, List[Path]] = {}
    for sid, paths in src_duplicates.items():
        all_duplicates.setdefault(sid, []).extend(paths)
    for sid, paths in tgt_duplicates.items():
        all_duplicates.setdefault(sid, []).extend(paths)
    for sid in list(all_duplicates.keys()):
        all_duplicates[sid] = sorted(set(all_duplicates[sid]), key=lambda p: str(p))
    
    # Validate each ID
    samples = []
    missing_src = []
    missing_tgt_image = []
    missing_tgt_txt = []
    warnings = []
    errors = []
    
    for sid, paths in sorted(tgt_txt_duplicates.items(), key=lambda kv: kv[0]):
        chosen = tgt_txt_map.get(sid)
        chosen_str = str(chosen) if chosen else "(none)"
        warnings.append(
            f"ID {sid}: multiple .txt files found ({len(paths)}). Using: {chosen_str}"
        )
    
    for sample_id in sorted(all_ids):
        # Skip IDs with duplicates
        if sample_id in all_duplicates:
            continue
        
        has_src = sample_id in src_map
        has_tgt_img = sample_id in tgt_img_map
        has_tgt_txt = sample_id in tgt_txt_map
        
        if not has_src:
            missing_src.append(sample_id)
            continue
        
        if not has_tgt_img:
            missing_tgt_image.append(sample_id)
            continue
        
        if not has_tgt_txt:
            missing_tgt_txt.append(sample_id)
            continue
        
        # Try to read prompt
        try:
            prompt = read_prompt_txt(tgt_txt_map[sample_id])
        except UnicodeDecodeError as e:
            errors.append(f"ID {sample_id}: {e}")
            continue
        except Exception as e:
            errors.append(f"ID {sample_id}: Error reading txt: {e}")
            continue
        
        samples.append({
            "id": sample_id,
            "src_image": src_map[sample_id],
            "tgt_image": tgt_img_map[sample_id],
            "tgt_txt": tgt_txt_map[sample_id],
            "prompt": prompt,
        })
    
    return ScanResult(
        samples=samples,
        missing_src=missing_src,
        missing_tgt_image=missing_tgt_image,
        missing_tgt_txt=missing_tgt_txt,
        duplicate_ids=all_duplicates,
        warnings=warnings,
        errors=errors,
    )


def print_scan_report(result: ScanResult, verbose: bool = False) -> None:
    """Print a human-readable scan report."""
    print(f"\n{'='*60}")
    print("SCAN REPORT")
    print(f"{'='*60}")
    
    print(f"\nTotal unique IDs found: {result.num_total_ids}")
    print(f"Valid samples: {result.num_valid_samples}")
    
    if result.has_issues:
        print(f"\n⚠️  ISSUES FOUND:")
        
        if result.duplicate_ids:
            print(f"\n  Duplicate IDs (same stem, multiple images):")
            for sid, paths in result.duplicate_ids.items():
                print(f"    - {sid}: {', '.join(str(p) for p in paths)}")
        
        if result.missing_src:
            print(f"\n  Missing in source dir (A/): {len(result.missing_src)}")
            if verbose:
                for sid in result.missing_src[:10]:
                    print(f"    - {sid}")
                if len(result.missing_src) > 10:
                    print(f"    ... and {len(result.missing_src) - 10} more")
        
        if result.missing_tgt_image:
            print(f"\n  Missing target image (B/): {len(result.missing_tgt_image)}")
            if verbose:
                for sid in result.missing_tgt_image[:10]:
                    print(f"    - {sid}")
                if len(result.missing_tgt_image) > 10:
                    print(f"    ... and {len(result.missing_tgt_image) - 10} more")
        
        if result.missing_tgt_txt:
            print(f"\n  Missing txt file (B/): {len(result.missing_tgt_txt)}")
            if verbose:
                for sid in result.missing_tgt_txt[:10]:
                    print(f"    - {sid}")
                if len(result.missing_tgt_txt) > 10:
                    print(f"    ... and {len(result.missing_tgt_txt) - 10} more")
        
        if result.errors:
            print(f"\n  Other errors: {len(result.errors)}")
            for err in result.errors[:10]:
                print(f"    - {err}")
            if len(result.errors) > 10:
                print(f"    ... and {len(result.errors) - 10} more")
        
        if result.warnings:
            print(f"\n  Warnings: {len(result.warnings)}")
            for w in result.warnings[:10]:
                print(f"    - {w}")
            if len(result.warnings) > 10:
                print(f"    ... and {len(result.warnings) - 10} more")
    elif result.warnings:
        print(f"\n⚠️  WARNINGS:")
        for w in result.warnings[:10]:
            print(f"    - {w}")
        if len(result.warnings) > 10:
            print(f"    ... and {len(result.warnings) - 10} more")
        print("\n✅ Samples are usable (warnings above).")
    else:
        print("\n✅ All samples are valid!")
    
    print(f"\n{'='*60}")


def get_exit_code(result: ScanResult) -> int:
    """Get exit code based on scan result."""
    return 2 if result.has_issues else 0
