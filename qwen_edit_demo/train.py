"""Training backend wrapper for LoRA training."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional


SUPPORTED_BACKENDS = {"diffsynth", "auto"}


def is_qwen_2511_model(base_model: str) -> bool:
    """Check if the base model is Qwen-Image-Edit-2511."""
    return "Edit-2511" in base_model or "2511" in base_model


def build_diffsynth_train_command(
    dataset_dir: Path,
    base_model: str,
    output_dir: Path,
    max_steps: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    resolution: int,
    mixed_precision: str,
    num_workers: int,
    zero_cond_t: bool,
    diffsynth_repo: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Build the training command for DiffSynth-Studio backend.
    
    Args:
        dataset_dir: Path to prepared dataset
        base_model: Base model name or path
        output_dir: Output directory for training results
        max_steps: Maximum training steps
        lr: Learning rate
        batch_size: Batch size per device
        grad_accum: Gradient accumulation steps
        resolution: Training resolution
        mixed_precision: Mixed precision mode (fp16, bf16, etc.)
        num_workers: Number of data loading workers
        zero_cond_t: Whether to use zero conditioning for timestep (2511 specific)
        diffsynth_repo: Path to DiffSynth-Studio repository
        extra_args: Additional arguments to pass to training script
    
    Returns:
        Command as list of strings
    """
    # Determine training script path
    if diffsynth_repo:
        train_script = Path(diffsynth_repo) / "examples" / "qwen_image" / "train_qwen_image_lora.py"
        if not train_script.exists():
            # Try alternative paths
            train_script = Path(diffsynth_repo) / "train_qwen_image_lora.py"
    else:
        # Assume script is in PATH or current directory
        train_script = Path("train_qwen_image_lora.py")
    
    # Build command
    cmd = [
        sys.executable,
        str(train_script),
        "--pretrained_model_name_or_path", base_model,
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--max_train_steps", str(max_steps),
        "--learning_rate", str(lr),
        "--train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--resolution", str(resolution),
        "--mixed_precision", mixed_precision,
        "--dataloader_num_workers", str(num_workers),
    ]
    
    # Add 2511-specific zero_cond_t flag
    if zero_cond_t:
        cmd.append("--zero_cond_t")
    
    # Add extra args
    if extra_args:
        cmd.extend(extra_args)
    
    return cmd


def build_accelerate_command(
    train_cmd: List[str],
    num_processes: int = 1,
    mixed_precision: str = "bf16",
) -> List[str]:
    """Wrap training command with accelerate launch."""
    accel_cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "--mixed_precision", mixed_precision,
    ]
    accel_cmd.extend(train_cmd[1:])  # Skip the python executable
    return accel_cmd


def train(
    dataset_dir: Path,
    output_dir: Path,
    backend: str = "diffsynth",
    base_model: str = "Qwen/Qwen-Image-Edit-2511",
    max_steps: int = 2000,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    resolution: int = 512,
    mixed_precision: str = "bf16",
    num_workers: int = 4,
    zero_cond_t: Optional[bool] = None,
    diffsynth_repo: Optional[Path] = None,
    use_accelerate: bool = True,
    num_processes: int = 1,
    dry_run: bool = False,
) -> Dict:
    """
    Launch LoRA training with specified backend.
    
    Args:
        dataset_dir: Path to prepared dataset
        output_dir: Output directory for training results
        backend: Training backend ("diffsynth" or "auto")
        base_model: Base model name or path
        max_steps: Maximum training steps
        lr: Learning rate
        batch_size: Batch size per device
        grad_accum: Gradient accumulation steps
        resolution: Training resolution
        mixed_precision: Mixed precision mode
        num_workers: Number of data loading workers
        zero_cond_t: Whether to use zero conditioning for timestep (2511 specific).
                     If None, auto-detects based on base_model name.
        diffsynth_repo: Path to DiffSynth-Studio repository
        use_accelerate: Whether to use accelerate launch
        num_processes: Number of processes for accelerate
        dry_run: If True, only print command without executing
    
    Returns:
        Dict with training results and paths
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Choose from {SUPPORTED_BACKENDS}")
    
    # Auto-select backend if needed
    if backend == "auto":
        backend = "diffsynth"
    
    # Determine zero_cond_t default based on model
    is_2511 = is_qwen_2511_model(base_model)
    if zero_cond_t is None:
        zero_cond_t = is_2511
    
    # Warning for 2511 without zero_cond_t
    if is_2511 and not zero_cond_t:
        warnings.warn(
            f"⚠️  WARNING: You are training Qwen-Image-Edit-2511 without --zero_cond_t. "
            f"This parameter is strongly recommended for 2511 models. "
            f"See: https://github.com/ostris/ai-toolkit/issues/602",
            UserWarning,
            stacklevel=2
        )
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    if backend == "diffsynth":
        cmd = build_diffsynth_train_command(
            dataset_dir=dataset_dir,
            base_model=base_model,
            output_dir=output_dir,
            max_steps=max_steps,
            lr=lr,
            batch_size=batch_size,
            grad_accum=grad_accum,
            resolution=resolution,
            mixed_precision=mixed_precision,
            num_workers=num_workers,
            zero_cond_t=zero_cond_t,
            diffsynth_repo=diffsynth_repo,
        )
        
        if use_accelerate:
            cmd = build_accelerate_command(cmd, num_processes, mixed_precision)
    else:
        raise ValueError(f"Backend '{backend}' not yet implemented")
    
    # Save configuration
    config = {
        "backend": backend,
        "base_model": base_model,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "max_steps": max_steps,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "resolution": resolution,
        "mixed_precision": mixed_precision,
        "num_workers": num_workers,
        "zero_cond_t": zero_cond_t,
        "is_2511": is_2511,
        "use_accelerate": use_accelerate,
        "num_processes": num_processes,
    }
    
    config_path = output_dir / "config.resolved.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Save command to cmd.sh
    cmd_path = output_dir / "cmd.sh"
    cmd_str = " \\\n  ".join(cmd)
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write(cmd_str + "\n")
    os.chmod(cmd_path, 0o755)
    
    # Setup logging
    log_path = output_dir / "train.log"
    
    if dry_run:
        print("Dry run mode - command that would be executed:")
        print(cmd_str)
        return {
            "success": True,
            "dry_run": True,
            "command": cmd,
            "config_path": config_path,
            "cmd_path": cmd_path,
        }
    
    # Run training
    print(f"Starting training with backend: {backend}")
    print(f"Base model: {base_model}")
    print(f"Output dir: {output_dir}")
    print(f"Zero cond t: {zero_cond_t} {'(2511 recommended)' if is_2511 else ''}")
    print(f"Command: {cmd_str[:200]}...")
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# Training started at {__import__('datetime').datetime.now().isoformat()}\n")
        log_file.write(f"# Command: {cmd_str}\n\n")
        log_file.flush()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\n❌ Training failed with exit code: {process.returncode}")
                return {
                    "success": False,
                    "returncode": process.returncode,
                    "log_path": log_path,
                    "config_path": config_path,
                }
            
            print(f"\n✅ Training completed successfully!")
            return {
                "success": True,
                "returncode": 0,
                "log_path": log_path,
                "config_path": config_path,
                "cmd_path": cmd_path,
                "output_dir": output_dir,
            }
            
        except FileNotFoundError as e:
            error_msg = f"Training script not found: {e}"
            print(f"\n❌ {error_msg}")
            log_file.write(f"\nERROR: {error_msg}\n")
            return {
                "success": False,
                "error": error_msg,
                "log_path": log_path,
            }
        except Exception as e:
            error_msg = f"Training error: {e}"
            print(f"\n❌ {error_msg}")
            log_file.write(f"\nERROR: {error_msg}\n")
            return {
                "success": False,
                "error": str(e),
                "log_path": log_path,
            }


def find_lora_weights(output_dir: Path) -> Optional[Path]:
    """
    Find LoRA weights in output directory.
    
    Returns:
        Path to LoRA weights file, or None if not found
    """
    output_dir = Path(output_dir)
    
    # Common patterns
    patterns = [
        "lora.safetensors",
        "pytorch_lora_weights.safetensors",
        "lora/pytorch_lora_weights.safetensors",
        "checkpoint-*/pytorch_lora_weights.safetensors",
    ]
    
    for pattern in patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Return the most recent one
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None
