# qwen_edit_demo

一个 Python CLI：把 (原图 A/) + (目标图+指令 B/) 转成 Qwen-Image-Edit LoRA 训练可用的数据清单，并支持 LoRA 推理验收。

## 功能

- **inspect**: 扫描并校验数据配对是否完整
- **prepare**: 生成标准化训练数据目录 + metadata.jsonl
- **train**: 启动 LoRA 训练（支持 DiffSynth-Studio 后端）
- **infer**: 加载 LoRA 做推理验收
- **report**: 生成验收报告（HTML/Markdown）

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd qwen_edit_demo

# 安装依赖
pip install -r requirements.txt

# 注意：diffusers 可能需要从源码安装以支持 QwenImage
pip install git+https://github.com/huggingface/diffusers.git
```

## Quickstart

### 1) 准备数据

确保你的数据符合以下结构：

```
A_src/                       # 原图文件夹
  0001.png
  0002.jpg
  ...

B_tgt/                       # 修改后文件夹
  0001.png                   # 目标图
  0001.txt                   # 指令文本
  0002.jpg
  0002.txt
  ...
```

### 2) 检查数据

```bash
python -m qwen_edit_demo inspect \
  --src_dir A_src \
  --tgt_dir B_tgt
```

### 3) 准备数据集

```bash
python -m qwen_edit_demo prepare \
  --src_dir A_src \
  --tgt_dir B_tgt \
  --out_dir out_dataset \
  --val_ratio 0.05
```

### 4) 训练 LoRA

```bash
python -m qwen_edit_demo train \
  --dataset_dir out_dataset \
  --backend diffsynth \
  --base_model Qwen/Qwen-Image-Edit-2511 \
  --output_dir runs/exp001 \
  --max_steps 2000 \
  --lr 1e-4 \
  --zero_cond_t 1
```

### 5) 推理验收

```bash
python -m qwen_edit_demo infer \
  --dataset_dir out_dataset \
  --split val \
  --base_model Qwen/Qwen-Image-Edit-2511 \
  --lora_path runs/exp001/lora.safetensors \
  --out_dir infer_out/exp001_val
```

### 6) 生成报告

```bash
python -m qwen_edit_demo report \
  --infer_dir infer_out/exp001_val \
  --format html
```

## 关键坑位提示

1. **`--zero_cond_t`**: 2511 LoRA 训练建议开启，Demo 默认强制为 1。
2. **multi-image 输入**: 2511 单图也用 list。
3. **CFG**: 用 `true_cfg_scale`，并传 `negative_prompt`（哪怕 `" "`）。
4. **diffusers 版本**: 可能需要从 main 分支安装。
5. 数据里若存在同 stem 多扩展名文件（`0001.png` + `0001.jpg`），默认直接报错。

## 项目结构

```
qwen_edit_demo/
  __init__.py
  __main__.py          # python -m qwen_edit_demo
  cli.py               # Typer CLI 接口
  scan.py              # 扫描配对逻辑
  prepare.py           # copy/link + metadata.jsonl
  train.py             # 训练后端封装
  infer.py             # diffusers 推理
  report.py            # 报告生成
  utils.py             # 工具函数
```

## License

MIT
