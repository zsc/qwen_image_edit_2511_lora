"""Inference with trained LoRA using diffusers."""

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm


def load_pipeline(
    base_model: str,
    lora_path: Optional[Path] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Load QwenImageEditPipeline with optional LoRA weights.
    
    Args:
        base_model: Base model name or path
        lora_path: Path to LoRA weights
        torch_dtype: Torch data type
        device: Device to load model on
    
    Returns:
        Loaded pipeline
    """
    try:
        from diffusers import QwenImageEditPipeline
    except ImportError:
        raise ImportError(
            "diffusers not installed or QwenImageEditPipeline not available. "
            "Please install diffusers from source: "
            "pip install git+https://github.com/huggingface/diffusers.git"
        )
    
    print(f"Loading base model: {base_model}")
    pipe = QwenImageEditPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
    )
    
    if lora_path:
        lora_path = Path(lora_path)
        print(f"Loading LoRA weights: {lora_path}")
        if lora_path.exists():
            pipe.load_lora_weights(str(lora_path))
        else:
            warnings.warn(f"LoRA path not found: {lora_path}")
    
    pipe = pipe.to(device)
    return pipe


def run_inference(
    pipe,
    image: Union[Image.Image, List[Image.Image]],
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    negative_prompt: str = " ",
    true_cfg_scale: Optional[float] = None,
    seed: int = 0,
) -> Image.Image:
    """
    Run inference on a single image.
    
    Args:
        pipe: Loaded pipeline
        image: Input image(s). For 2511, pass as list even for single image.
        prompt: Edit prompt
        height: Output height
        width: Output width
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        negative_prompt: Negative prompt (required for CFG, use " " to enable)
        true_cfg_scale: True CFG scale (for 2511)
        seed: Random seed
    
    Returns:
        Generated image
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Prepare kwargs
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    
    # For 2511: use true_cfg_scale and negative_prompt for CFG
    if true_cfg_scale is not None:
        kwargs["true_cfg_scale"] = true_cfg_scale
    
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    
    # For Qwen-Image-Edit-2511: pass image as list even for single image
    # See: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/
    if not isinstance(image, list):
        image = [image]
    
    kwargs["image"] = image
    
    result = pipe(**kwargs)
    return result.images[0]


def infer_dataset(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "Qwen/Qwen-Image-Edit-2511",
    lora_path: Optional[Path] = None,
    split: str = "val",
    num_inference_steps: int = 20,
    true_cfg_scale: float = 4.0,
    negative_prompt: str = " ",
    height: int = 512,
    width: int = 512,
    seed: int = 0,
    torch_dtype: str = "bfloat16",
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Run inference on a prepared dataset.
    
    Args:
        dataset_dir: Path to prepared dataset
        output_dir: Output directory for inference results
        base_model: Base model name or path
        lora_path: Path to LoRA weights
        split: Which split to use ("train", "val", or "all")
        num_inference_steps: Number of denoising steps
        true_cfg_scale: True CFG scale for 2511
        negative_prompt: Negative prompt for CFG
        height: Output height
        width: Output width
        seed: Random seed
        torch_dtype: Torch dtype string
        device: Device to use
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        Dict with inference results
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_val = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Load metadata
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {dataset_dir}")
    
    all_samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_samples.append(json.loads(line))
    
    # Filter by split if specified
    if split != "all":
        split_path = dataset_dir / "split.json"
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            split_ids = set(split_data.get(split, []))
            samples = [s for s in all_samples if s["id"] in split_ids]
        else:
            # No split file, use all samples
            samples = all_samples
    else:
        samples = all_samples
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Running inference on {len(samples)} samples (split: {split})")
    
    # Load pipeline
    pipe = load_pipeline(base_model, lora_path, torch_dtype_val, device)
    
    # Check if 2511 model
    is_2511 = "Edit-2511" in base_model or "2511" in base_model
    if is_2511:
        print("Detected 2511 model - using list format for images")
    
    # Process samples
    results = []
    errors = []
    
    for sample in tqdm(samples, desc="Inference"):
        sample_id = sample["id"]
        sample_out_dir = output_dir / sample_id
        sample_out_dir.mkdir(exist_ok=True)
        
        try:
            # Load control image
            control_rel = sample["control_image"]
            control_path = dataset_dir / control_rel
            if not control_path.exists():
                raise FileNotFoundError(f"Control image not found: {control_path}")
            
            control_img = Image.open(control_path).convert("RGB")
            
            # Load target image for reference
            target_rel = sample["image"]
            target_path = dataset_dir / target_rel
            target_img = None
            if target_path.exists():
                target_img = Image.open(target_path).convert("RGB")
            
            prompt = sample["prompt"]
            
            # Save control and target
            control_img.save(sample_out_dir / "control.png")
            if target_img:
                target_img.save(sample_out_dir / "target.png")
            
            # Save prompt
            with open(sample_out_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            # Run inference
            pred_img = run_inference(
                pipe=pipe,
                image=control_img,  # Will be converted to list internally
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                seed=seed,
            )
            
            # Save prediction
            pred_img.save(sample_out_dir / "pred.png")
            
            results.append({
                "id": sample_id,
                "prompt": prompt,
                "success": True,
                "output_dir": str(sample_out_dir),
            })
            
        except Exception as e:
            error_msg = f"Error processing {sample_id}: {e}"
            print(error_msg)
            errors.append({
                "id": sample_id,
                "error": str(e),
            })
            
            # Write error file
            with open(sample_out_dir / "error.txt", "w", encoding="utf-8") as f:
                f.write(error_msg)
    
    # Write index.csv
    index_path = output_dir / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "control_path", "target_path", "pred_path", "status"])
        for r in results:
            sample_dir = output_dir / r["id"]
            writer.writerow([
                r["id"],
                r["prompt"],
                str(sample_dir / "control.png"),
                str(sample_dir / "target.png"),
                str(sample_dir / "pred.png"),
                "success",
            ])
        for e in errors:
            writer.writerow([
                e["id"],
                "",
                "",
                "",
                "",
                f"error: {e['error']}",
            ])
    
    # Create summary
    summary = {
        "total_samples": len(samples),
        "successful": len(results),
        "failed": len(errors),
        "output_dir": str(output_dir),
        "base_model": base_model,
        "lora_path": str(lora_path) if lora_path else None,
    }
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Inference complete!")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {len(errors)}")
    print(f"   Output: {output_dir}")
    
    return summary


def create_contact_sheet(
    infer_dir: Path,
    output_path: Optional[Path] = None,
    samples_per_row: int = 4,
    thumb_size: int = 256,
) -> Optional[Path]:
    """
    Create a contact sheet image from inference results.
    
    Args:
        infer_dir: Inference output directory
        output_path: Output path for contact sheet (default: infer_dir/contact_sheet.png)
        samples_per_row: Number of samples per row
        thumb_size: Thumbnail size
    
    Returns:
        Path to created contact sheet, or None if no images found.
    """
    infer_dir = Path(infer_dir)
    if output_path is None:
        output_path = infer_dir / "contact_sheet.png"
    
    # Find all sample directories
    sample_dirs = sorted([d for d in infer_dir.iterdir() if d.is_dir()])
    
    images_data = []
    for sample_dir in sample_dirs:
        control_path = sample_dir / "control.png"
        target_path = sample_dir / "target.png"
        pred_path = sample_dir / "pred.png"
        prompt_path = sample_dir / "prompt.txt"
        
        if pred_path.exists():
            data = {
                "id": sample_dir.name,
                "pred": Image.open(pred_path) if pred_path.exists() else None,
                "control": Image.open(control_path) if control_path.exists() else None,
                "target": Image.open(target_path) if target_path.exists() else None,
            }
            
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    data["prompt"] = f.read().strip()
            else:
                data["prompt"] = ""
            
            images_data.append(data)
    
    if not images_data:
        print("No images found for contact sheet")
        return None
    
    # Calculate grid size
    num_samples = len(images_data)
    num_cols = min(samples_per_row, num_samples)
    num_rows = (num_samples + num_cols - 1) // num_cols
    
    # Each sample shows: control | target | pred (3 images side by side)
    cell_width = thumb_size
    cell_height = thumb_size
    label_height = 30
    
    img_width = num_cols * (cell_width * 3 + 20) + 20
    img_height = num_rows * (cell_height + label_height + 20) + 20
    
    # Create canvas
    canvas = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
    except:
        draw = None
        font = None
    
    for idx, data in enumerate(images_data):
        row = idx // num_cols
        col = idx % num_cols
        
        x_offset = 20 + col * (cell_width * 3 + 20)
        y_offset = 20 + row * (cell_height + label_height + 20)
        
        # Paste images: control | target | pred
        imgs_to_paste = [
            (data.get("control"), "Input"),
            (data.get("target"), "Target"),
            (data.get("pred"), "Output"),
        ]
        
        for i, (img, label) in enumerate(imgs_to_paste):
            if img:
                img_thumb = img.copy()
                img_thumb.thumbnail((cell_width, cell_height))
                
                paste_x = x_offset + i * cell_width + (cell_width - img_thumb.width) // 2
                paste_y = y_offset + (cell_height - img_thumb.height) // 2
                
                canvas.paste(img_thumb, (paste_x, paste_y))
        
        # Draw label
        if draw and font:
            label_text = f"{data['id']}: {data['prompt'][:40]}"
            draw.text((x_offset, y_offset + cell_height + 5), label_text, fill=(0, 0, 0), font=font)
    
    canvas.save(output_path)
    print(f"Contact sheet saved: {output_path}")
    return output_path
