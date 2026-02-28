"""CLI interface for qwen_edit_demo using Typer."""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from . import __version__
from .scan import scan_directory, print_scan_report, get_exit_code
from .prepare import prepare_dataset
from .train import train, find_lora_weights
from .infer import infer_dataset, create_contact_sheet
from .report import generate_markdown_report, generate_html_report


app = typer.Typer(
    name="qwen_edit_demo",
    help="Qwen Image Edit Demo - LoRA training data preparation and inference",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"qwen_edit_demo {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=version_callback, is_eager=True),
    ] = None,
):
    """Qwen Image Edit Demo - CLI for LoRA training and inference."""
    pass


@app.command()
def inspect(
    src_dir: Annotated[
        Path,
        typer.Option("--src_dir", help="Directory containing source images (A/)"),
    ],
    tgt_dir: Annotated[
        Path,
        typer.Option("--tgt_dir", help="Directory containing target images and txt files (B/)"),
    ],
    exts: Annotated[
        str,
        typer.Option("--exts", help="Comma-separated list of allowed extensions"),
    ] = "png,jpg,jpeg,webp",
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Scan subdirectories recursively"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
):
    """
    Inspect and validate image pairs for training data preparation.
    
    Checks:
    - Each ID has corresponding images in src_dir and tgt_dir
    - Each ID has a txt file in tgt_dir with the prompt
    - No duplicate IDs (same stem with different extensions)
    - txt files are valid UTF-8
    
    Exit codes:
    - 0: No issues found
    - 2: Issues detected (missing files, duplicates, etc.)
    """
    if not src_dir.exists():
        typer.echo(f"❌ Source directory not found: {src_dir}", err=True)
        raise typer.Exit(code=1)
    
    if not tgt_dir.exists():
        typer.echo(f"❌ Target directory not found: {tgt_dir}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Scanning directories...")
    typer.echo(f"  Source: {src_dir}")
    typer.echo(f"  Target: {tgt_dir}")
    typer.echo(f"  Extensions: {exts}")
    typer.echo(f"  Recursive: {recursive}")
    
    result = scan_directory(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        exts=exts,
        recursive=recursive,
    )
    
    print_scan_report(result, verbose=verbose)
    
    exit_code = get_exit_code(result)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
    
    typer.echo("\n✅ All checks passed!")


@app.command()
def prepare(
    src_dir: Annotated[
        Path,
        typer.Option("--src_dir", help="Directory containing source images (A/)"),
    ],
    tgt_dir: Annotated[
        Path,
        typer.Option("--tgt_dir", help="Directory containing target images and txt files (B/)"),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out_dir", help="Output directory for prepared dataset"),
    ],
    val_ratio: Annotated[
        float,
        typer.Option("--val_ratio", help="Validation set ratio (0-0.5)"),
    ] = 0.05,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for train/val split"),
    ] = 42,
    copy_mode: Annotated[
        str,
        typer.Option("--copy_mode", help="File copy mode: copy, hardlink, or symlink"),
    ] = "symlink",
    exts: Annotated[
        str,
        typer.Option("--exts", help="Comma-separated list of allowed extensions"),
    ] = "png,jpg,jpeg,webp",
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Scan subdirectories recursively"),
    ] = False,
):
    """
    Prepare training dataset from source and target directories.
    
    Creates a standardized dataset structure:
    - images/control/   - Source images
    - images/target/    - Target images
    - prompts/          - Prompt text files
    - metadata.jsonl    - Training manifest
    - manifest.json     - Dataset statistics
    - split.json        - Train/val split (if val_ratio > 0)
    """
    if not src_dir.exists():
        typer.echo(f"❌ Source directory not found: {src_dir}", err=True)
        raise typer.Exit(code=1)
    
    if not tgt_dir.exists():
        typer.echo(f"❌ Target directory not found: {tgt_dir}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Preparing dataset...")
    typer.echo(f"  Source: {src_dir}")
    typer.echo(f"  Target: {tgt_dir}")
    typer.echo(f"  Output: {out_dir}")
    typer.echo(f"  Copy mode: {copy_mode}")
    typer.echo(f"  Val ratio: {val_ratio}")
    
    # First scan
    scan_result = scan_directory(
        src_dir=src_dir,
        tgt_dir=tgt_dir,
        exts=exts,
        recursive=recursive,
    )
    
    if scan_result.has_issues:
        print_scan_report(scan_result, verbose=False)
        typer.echo("\n❌ Cannot prepare dataset - issues found. Run 'inspect' for details.", err=True)
        raise typer.Exit(code=2)
    
    if not scan_result.samples:
        typer.echo("❌ No valid samples found.", err=True)
        raise typer.Exit(code=1)
    
    # Prepare dataset
    manifest = prepare_dataset(
        samples=scan_result.samples,
        out_dir=out_dir,
        val_ratio=val_ratio,
        seed=seed,
        copy_mode=copy_mode,
        src_dir=src_dir,
        tgt_dir=tgt_dir,
    )
    
    typer.echo(f"\n✅ Dataset prepared successfully!")
    typer.echo(f"   Total samples: {manifest['num_total']}")
    typer.echo(f"   Train: {manifest['num_train']}")
    typer.echo(f"   Val: {manifest['num_val']}")
    typer.echo(f"   Output: {out_dir}")


@app.command(name="train")
def train_cmd(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset_dir", help="Path to prepared dataset"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output_dir", help="Output directory for training results"),
    ],
    backend: Annotated[
        str,
        typer.Option("--backend", help="Training backend: diffsynth or auto"),
    ] = "diffsynth",
    base_model: Annotated[
        str,
        typer.Option("--base_model", help="Base model name or path"),
    ] = "Qwen/Qwen-Image-Edit-2511",
    max_steps: Annotated[
        int,
        typer.Option("--max_steps", help="Maximum training steps"),
    ] = 2000,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", help="Batch size per device"),
    ] = 1,
    grad_accum: Annotated[
        int,
        typer.Option("--grad_accum", help="Gradient accumulation steps"),
    ] = 4,
    resolution: Annotated[
        int,
        typer.Option("--resolution", help="Training resolution"),
    ] = 512,
    mixed_precision: Annotated[
        str,
        typer.Option("--mixed_precision", help="Mixed precision: fp16, bf16, fp32"),
    ] = "bf16",
    num_workers: Annotated[
        int,
        typer.Option("--num_workers", help="Number of data loading workers"),
    ] = 4,
    zero_cond_t: Annotated[
        Optional[int],
        typer.Option("--zero_cond_t", help="Enable zero conditioning for 2511 (0 or 1, default: auto)"),
    ] = None,
    diffsynth_repo: Annotated[
        Optional[Path],
        typer.Option("--diffsynth_repo", help="Path to DiffSynth-Studio repository"),
    ] = None,
    use_accelerate: Annotated[
        bool,
        typer.Option("--use_accelerate", help="Use accelerate launch"),
    ] = True,
    num_processes: Annotated[
        int,
        typer.Option("--num_processes", help="Number of processes for accelerate"),
    ] = 1,
    dry_run: Annotated[
        bool,
        typer.Option("--dry_run", help="Print command without executing"),
    ] = False,
):
    """
    Launch LoRA training with specified backend.
    
    For Qwen-Image-Edit-2511, --zero_cond_t is strongly recommended and enabled by default.
    See: https://github.com/ostris/ai-toolkit/issues/602
    
    Output files:
    - lora.safetensors   - Trained LoRA weights
    - train.log          - Training log
    - config.resolved.json - Final configuration
    - cmd.sh             - Command used for training
    """
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset directory not found: {dataset_dir}", err=True)
        raise typer.Exit(code=1)
    
    # Convert zero_cond_t to bool
    zero_cond_t_bool = None if zero_cond_t is None else bool(zero_cond_t)
    
    result = train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        backend=backend,
        base_model=base_model,
        max_steps=max_steps,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        resolution=resolution,
        mixed_precision=mixed_precision,
        num_workers=num_workers,
        zero_cond_t=zero_cond_t_bool,
        diffsynth_repo=diffsynth_repo,
        use_accelerate=use_accelerate,
        num_processes=num_processes,
        dry_run=dry_run,
    )
    
    if dry_run:
        return
    
    if not result.get("success"):
        raise typer.Exit(code=1)
    
    # Check for LoRA weights
    lora_path = find_lora_weights(output_dir)
    if lora_path:
        typer.echo(f"\n🎉 LoRA weights: {lora_path}")


@app.command()
def infer(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset_dir", help="Path to prepared dataset"),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out_dir", help="Output directory for inference results"),
    ],
    base_model: Annotated[
        str,
        typer.Option("--base_model", help="Base model name or path"),
    ] = "Qwen/Qwen-Image-Edit-2511",
    lora_path: Annotated[
        Optional[Path],
        typer.Option("--lora_path", help="Path to LoRA weights"),
    ] = None,
    split: Annotated[
        str,
        typer.Option("--split", help="Which split to use: train, val, or all"),
    ] = "val",
    num_inference_steps: Annotated[
        int,
        typer.Option("--num_inference_steps", help="Number of denoising steps"),
    ] = 20,
    true_cfg_scale: Annotated[
        float,
        typer.Option("--true_cfg_scale", help="True CFG scale for 2511"),
    ] = 4.0,
    negative_prompt: Annotated[
        str,
        typer.Option("--negative_prompt", help="Negative prompt for CFG"),
    ] = " ",
    height: Annotated[
        int,
        typer.Option("--height", help="Output height"),
    ] = 512,
    width: Annotated[
        int,
        typer.Option("--width", help="Output width"),
    ] = 512,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed"),
    ] = 0,
    torch_dtype: Annotated[
        str,
        typer.Option("--torch_dtype", help="Torch dtype: float32, float16, bfloat16"),
    ] = "bfloat16",
    device: Annotated[
        str,
        typer.Option("--device", help="Device to use"),
    ] = "cuda",
    max_samples: Annotated[
        Optional[int],
        typer.Option("--max_samples", help="Maximum number of samples to process"),
    ] = None,
    contact_sheet: Annotated[
        bool,
        typer.Option("--contact_sheet", help="Generate contact sheet"),
    ] = True,
):
    """
    Run inference on validation set with trained LoRA.
    
    Uses diffusers QwenImageEditPipeline with LoRA weights.
    For 2511 models, images are passed as list even for single images.
    CFG is enabled via true_cfg_scale + negative_prompt.
    
    Output structure:
    - {sample_id}/
      - control.png  - Input image
      - target.png   - Target image
      - pred.png     - Model prediction
      - prompt.txt   - Prompt text
    - index.csv      - Summary of all results
    - summary.json   - Statistics
    """
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset directory not found: {dataset_dir}", err=True)
        raise typer.Exit(code=1)
    
    if lora_path and not lora_path.exists():
        typer.echo(f"⚠️  LoRA path not found: {lora_path}", err=True)
    
    summary = infer_dataset(
        dataset_dir=dataset_dir,
        output_dir=out_dir,
        base_model=base_model,
        lora_path=lora_path,
        split=split,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        seed=seed,
        torch_dtype=torch_dtype,
        device=device,
        max_samples=max_samples,
    )
    
    if contact_sheet and summary.get("successful", 0) > 0:
        typer.echo("\nGenerating contact sheet...")
        create_contact_sheet(out_dir)


@app.command()
def report(
    infer_dir: Annotated[
        Path,
        typer.Option("--infer_dir", help="Inference output directory"),
    ],
    out_file: Annotated[
        Optional[Path],
        typer.Option("--out_file", help="Output file path (default: infer_dir/report.md)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", help="Report format: markdown or html"),
    ] = "markdown",
    max_samples: Annotated[
        int,
        typer.Option("--max_samples", help="Maximum samples to include"),
    ] = 100,
):
    """
    Generate a report from inference results.
    
    Creates a human-readable report with:
    - Summary statistics
    - Sample comparisons (input, target, output)
    - Error details
    """
    if not infer_dir.exists():
        typer.echo(f"❌ Inference directory not found: {infer_dir}", err=True)
        raise typer.Exit(code=1)
    
    if format.lower() in ("markdown", "md"):
        output_path = generate_markdown_report(infer_dir, out_file, max_samples)
    elif format.lower() == "html":
        output_path = generate_html_report(infer_dir, out_file, max_samples)
    else:
        typer.echo(f"❌ Unknown format: {format}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"\n✅ Report generated: {output_path}")


def main():
    """Entry point for the CLI."""
    app()
