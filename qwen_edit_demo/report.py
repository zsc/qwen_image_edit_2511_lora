"""Generate reports from inference results."""

import csv
from pathlib import Path
from typing import Dict, List, Optional


def generate_markdown_report(
    infer_dir: Path,
    output_path: Optional[Path] = None,
    max_samples: int = 100,
) -> Path:
    """
    Generate a Markdown report from inference results.
    
    Args:
        infer_dir: Inference output directory
        output_path: Output path for report (default: infer_dir/report.md)
        max_samples: Maximum number of samples to include
    
    Returns:
        Path to generated report
    """
    infer_dir = Path(infer_dir)
    if output_path is None:
        output_path = infer_dir / "report.md"
    
    # Load summary
    summary_path = infer_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    
    # Load index
    index_path = infer_dir / "index.csv"
    rows = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    
    # Generate report
    lines = []
    lines.append("# Inference Report\n")
    
    # Summary section
    lines.append("## Summary\n")
    lines.append(f"- **Total Samples**: {summary.get('total_samples', len(rows))}")
    lines.append(f"- **Successful**: {summary.get('successful', 0)}")
    lines.append(f"- **Failed**: {summary.get('failed', 0)}")
    lines.append(f"- **Base Model**: {summary.get('base_model', 'N/A')}")
    lines.append(f"- **LoRA Path**: {summary.get('lora_path', 'N/A')}")
    lines.append(f"- **Output Directory**: {infer_dir}")
    lines.append("")
    
    # Results table
    lines.append("## Results\n")
    lines.append("| ID | Prompt | Input | Target | Output | Status |")
    lines.append("|---|---|---|---|---|---|")
    
    # Filter successful rows
    success_rows = [r for r in rows if r.get("status") == "success"]
    error_rows = [r for r in rows if r.get("status") != "success"]
    
    for row in success_rows[:max_samples]:
        sample_id = row.get("id", "")
        prompt = row.get("prompt", "").replace("|", "\\|")[:50]
        
        # Use relative paths for images
        sample_dir = infer_dir / sample_id
        control_rel = f"{sample_id}/control.png"
        target_rel = f"{sample_id}/target.png"
        pred_rel = f"{sample_id}/pred.png"
        
        lines.append(f"| {sample_id} | {prompt}... | ![Input]({control_rel}) | ![Target]({target_rel}) | ![Output]({pred_rel}) | ✅ |")
    
    # Show errors
    for row in error_rows[:20]:
        sample_id = row.get("id", "")
        status = row.get("status", "error")
        lines.append(f"| {sample_id} | - | - | - | - | ❌ {status} |")
    
    lines.append("")
    
    if len(success_rows) > max_samples:
        lines.append(f"\n*... and {len(success_rows) - max_samples} more samples*\n")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved: {output_path}")
    return output_path


def generate_html_report(
    infer_dir: Path,
    output_path: Optional[Path] = None,
    max_samples: int = 100,
) -> Path:
    """
    Generate an HTML report from inference results.
    
    Args:
        infer_dir: Inference output directory
        output_path: Output path for report (default: infer_dir/report.html)
        max_samples: Maximum number of samples to include
    
    Returns:
        Path to generated report
    """
    infer_dir = Path(infer_dir)
    if output_path is None:
        output_path = infer_dir / "report.html"
    
    # Load summary
    summary_path = infer_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    
    # Load index
    index_path = infer_dir / "index.csv"
    rows = []
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    
    # Filter rows
    success_rows = [r for r in rows if r.get("status") == "success"]
    error_rows = [r for r in rows if r.get("status") != "success"]
    
    # Build HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inference Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1, h2 { color: #333; }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .summary-item {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .summary-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .summary-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .sample-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #555;
        }
        .sample-prompt {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            font-style: italic;
        }
        .image-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .image-cell {
            text-align: center;
        }
        .image-cell img {
            width: 100%;
            max-height: 150px;
            object-fit: contain;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .image-label {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        .error-section {
            background: #fee;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .error-item {
            padding: 10px;
            background: white;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 4px solid #c00;
        }
    </style>
</head>
<body>
""")
    
    html_parts.append("<h1>🔮 Inference Report</h1>")
    
    # Summary section
    html_parts.append("<div class='summary'>")
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<div class='summary-grid'>")
    html_parts.append(f"<div class='summary-item'><div class='summary-label'>Total Samples</div><div class='summary-value'>{summary.get('total_samples', len(rows))}</div></div>")
    html_parts.append(f"<div class='summary-item'><div class='summary-label'>Successful</div><div class='summary-value' style='color: #28a745'>{summary.get('successful', 0)}</div></div>")
    html_parts.append(f"<div class='summary-item'><div class='summary-label'>Failed</div><div class='summary-value' style='color: #dc3545'>{summary.get('failed', 0)}</div></div>")
    html_parts.append("</div>")
    html_parts.append(f"<p style='margin-top: 15px; color: #666;'>Base Model: {summary.get('base_model', 'N/A')}</p>")
    html_parts.append("</div>")
    
    # Results section
    html_parts.append("<h2>Results</h2>")
    html_parts.append("<div class='sample-grid'>")
    
    for row in success_rows[:max_samples]:
        sample_id = row.get("id", "")
        prompt = row.get("prompt", "")
        
        html_parts.append(f"<div class='sample-card'>")
        html_parts.append(f"<div class='sample-header'>ID: {sample_id}</div>")
        html_parts.append(f"<div class='sample-prompt'>\"{prompt}\"</div>")
        html_parts.append("<div class='image-row'>")
        
        # Control image
        control_path = f"{sample_id}/control.png"
        html_parts.append(f"<div class='image-cell'>")
        html_parts.append(f"<img src='{control_path}' alt='Input' loading='lazy'>")
        html_parts.append("<div class='image-label'>Input</div>")
        html_parts.append("</div>")
        
        # Target image
        target_path = f"{sample_id}/target.png"
        html_parts.append(f"<div class='image-cell'>")
        html_parts.append(f"<img src='{target_path}' alt='Target' loading='lazy'>")
        html_parts.append("<div class='image-label'>Target</div>")
        html_parts.append("</div>")
        
        # Prediction
        pred_path = f"{sample_id}/pred.png"
        html_parts.append(f"<div class='image-cell'>")
        html_parts.append(f"<img src='{pred_path}' alt='Output' loading='lazy'>")
        html_parts.append("<div class='image-label'>Output</div>")
        html_parts.append("</div>")
        
        html_parts.append("</div>")
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Error section
    if error_rows:
        html_parts.append("<div class='error-section'>")
        html_parts.append("<h2>⚠️ Errors</h2>")
        for row in error_rows[:20]:
            sample_id = row.get("id", "unknown")
            status = row.get("status", "error")
            html_parts.append(f"<div class='error-item'><strong>{sample_id}</strong>: {status}</div>")
        html_parts.append("</div>")
    
    html_parts.append("</body></html>")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    
    print(f"HTML report saved: {output_path}")
    return output_path
