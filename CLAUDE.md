# Python 纯命令行 Demo SPEC（中文 Markdown）

> 目标：做一个**可直接交给 codex / gemini-cli 生成代码**的最小可用 CLI Demo，用两份文件夹（原图 / 修改后图+指令）快速产出训练数据清单，并提供 LoRA 训练“可插拔后端”的启动方式 + 推理验收。

---

## 0. 背景与关键约束（必须写进 Demo）

1. **Qwen-Image-Edit-2511 LoRA 训练需要 `--zero_cond_t`**（2511 专用参数，建议强制开启）。([GitHub][1])
2. **2511 是 multi-image editing**：即使只有一张输入图，也建议按“list 形式”喂给推理/训练代码（避免格式不匹配）。([GitHub][2])
3. 使用 `diffusers` 推理时：

   * `QwenImageEditPipeline.__call__` 的 `image` 参数支持 **单张或 list**。([Hugging Face][3])
   * 若要启用 CFG，需要传 `true_cfg_scale` + `negative_prompt`（哪怕是 `" "` 也能触发 CFG 计算）。([Hugging Face][3])
4. `diffusers` 的 QwenImage 文档页提示：你正在看的可能是 **main 分支文档**，可能需要从源码安装（`pip install git+...`）。([Hugging Face][3])

---

## 1. Demo 要解决什么问题（User Story）

用户手上只有两类素材：

* **原图文件夹**：`A/`（原图）
* **修改后文件夹**：`B/`（修改后的图 + 文本指令）

要求：

* 原图与修改后图 **同名/同 stem**（如 `0001.png` ↔ `0001.png`）
* 指令文本在修改后文件夹里：`0001.txt`（与图片同前缀/同 stem）

Demo 一键完成：

1. 扫描并校验配对是否完整
2. 生成标准化训练数据目录 + `metadata.jsonl`（或 csv）
3. 可选：调用一个“后端训练器”（如 DiffSynth-Studio / qwen-image-finetune / musubi-tuner）启动 LoRA 训练
4. 可选：加载 LoRA 做推理验收，输出对比图（原图 / 目标图 / 模型输出）

---

## 2. 输入数据约定（必须严格定义）

### 2.1 目录结构（最小约定）

```text
A_src/                       # 原图文件夹（控制图 / 输入图）
  0001.png
  0002.jpg
  ...

B_tgt/                       # 修改后文件夹（目标图 + 指令）
  0001.png                   # 目标图（编辑后）
  0001.txt                   # 指令（UTF-8 文本）
  0002.jpg
  0002.txt
  ...
```

### 2.2 配对规则（必须写清）

* 用“**stem**”作为 sample id：

  * `0001.png` 的 stem 是 `0001`
* 对每个 `id`：

  * 在 `A_src/` 必须存在一张图片
  * 在 `B_tgt/` 必须存在一张图片
  * 在 `B_tgt/` 必须存在 `id.txt`
* 允许图片扩展名：`.png .jpg .jpeg .webp`（可配置）
* `id.txt`：

  * UTF-8 读取
  * `strip()` 后如果为空：默认替换成 `"-"`（或报错，二选一；建议默认 `"-"`）

### 2.3 递归扫描（可选）

* 提供 `--recursive`：允许子目录
* 若递归：输出中仍建议扁平化（或保留相对路径；二选一，建议扁平化以减少后端兼容成本）

---

## 3. 输出数据格式（训练清单：metadata.jsonl）

### 3.1 统一采用三列语义（与业界 image editing 数据集一致）

参考常见 image-edit fine-tune 方式：

* `control_image`：输入/参考图（原图）
* `image`：目标图（编辑后）
* `prompt`：编辑指令

这种三列命名在 “图像编辑 fine-tune” 场景非常通用。([Oxen.ai][4])

### 3.2 Demo 输出目录（建议）

```text
out_dataset/
  images/
    control/                 # 原图（控制图）
      0001.png
      0002.jpg
    target/                  # 目标图（编辑后）
      0001.png
      0002.jpg
  prompts/
      0001.txt
      0002.txt
  metadata.jsonl             # 每行一个样本
  manifest.json              # 统计信息（数量、缺失、hash 可选）
  split.json                 # train/val 切分（可选）
```

### 3.3 metadata.jsonl 行格式（必须精确定义）

每行一个 JSON 对象（相对路径从 `out_dataset/` 开始）：

```json
{"id":"0001","control_image":"images/control/0001.png","image":"images/target/0001.png","prompt":"把背景改成雪山，保持人物不变"}
```

字段定义：

* `id`：字符串，sample id（stem）
* `control_image`：相对路径，指向原图
* `image`：相对路径，指向目标图
* `prompt`：编辑指令字符串

---

## 4. CLI 设计（核心：纯 Python 命令行）

实现方式建议：`typer` 或 `argparse`（二选一；建议 typer 体验更好）

入口命令统一为：

```bash
python -m qwen_edit_demo <subcommand> [options]
```

### 4.1 子命令一：`inspect`（只读校验）

用途：快速检查数据是否能训练

**命令：**

```bash
python -m qwen_edit_demo inspect \
  --src_dir /path/A_src \
  --tgt_dir /path/B_tgt \
  --exts png,jpg,jpeg,webp \
  --recursive 0
```

**输出：**

* stdout：打印统计（总数、成功配对数、缺失原图/缺失目标图/缺失 txt）
* exit code：

  * `0`：无缺失
  * `2`：存在缺失/重复/无法解析（用于 CI）

**必须检测的异常：**

* 同一个 `id` 出现多张图（例如 `0001.png` 和 `0001.jpg` 同时存在）→ 默认报错并列出冲突
* `id.txt` 不存在
* txt 不是 UTF-8（建议尝试 utf-8-sig；仍失败则报错）

---

### 4.2 子命令二：`prepare`（生成 out_dataset）

用途：把两文件夹数据“标准化”成训练可读格式

**命令：**

```bash
python -m qwen_edit_demo prepare \
  --src_dir /path/A_src \
  --tgt_dir /path/B_tgt \
  --out_dir /path/out_dataset \
  --val_ratio 0.05 \
  --seed 42 \
  --copy_mode symlink \
  --exts png,jpg,jpeg,webp \
  --recursive 0
```

**参数约束：**

* `--copy_mode` ∈ `{copy, hardlink, symlink}`

  * 默认 `symlink`（省空间；Windows 可回退 copy）
* `--val_ratio`：0~0.5；可为 0（不切分）
* `--seed`：用于 shuffle + split

**prepare 必须做的事：**

1. 复用 `inspect` 的扫描逻辑拿到 samples 列表（按 id）
2. 按 `copy_mode` 把文件放入：

   * `images/control/{id}.{ext}`
   * `images/target/{id}.{ext}`
   * `prompts/{id}.txt`
3. 生成 `metadata.jsonl`
4. 生成 `manifest.json`（示例字段建议如下）：

   ```json
   {
     "num_total": 5400,
     "num_train": 5130,
     "num_val": 270,
     "exts": ["png","jpg","jpeg","webp"],
     "created_at": "2026-02-02T00:00:00Z",
     "src_dir": "...",
     "tgt_dir": "..."
   }
   ```
5. 若 `val_ratio > 0`：写 `split.json`

   ```json
   {"train":["0001","0002",...], "val":["0101","0233",...]}
   ```

---

### 4.3 子命令三：`train`（可插拔后端：默认 diffsynth）

> 说明：这个 Demo 不重写训练框架，而是提供**统一的训练启动封装**。
> 你可以在实现时先支持 1 个后端（推荐 DiffSynth-Studio），其它后端留 TODO。

#### 4.3.1 train 通用接口（必须）

```bash
python -m qwen_edit_demo train \
  --dataset_dir /path/out_dataset \
  --backend diffsynth \
  --base_model Qwen/Qwen-Image-Edit-2511 \
  --output_dir /path/runs/exp001 \
  --max_steps 2000 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 4 \
  --resolution 512 \
  --mixed_precision bf16 \
  --num_workers 4 \
  --zero_cond_t 1
```

**硬性要求：**

* 当 `base_model` 包含 `Edit-2511` 且 `backend=difssynth`：

  * 默认 `--zero_cond_t=1`
  * 若用户显式传 `--zero_cond_t 0`：必须在 stdout 打印强 warning（但允许继续）
    依据：`--zero_cond_t` 被标注为 2511 专用参数，建议启用。([GitHub][1])

**输出要求：**

* `runs/exp001/`

  * `lora/` 或 `lora.safetensors`（按后端习惯）
  * `train.log`
  * `config.resolved.json`（把最终参数落盘，便于复现）
  * `cmd.sh`（把真正执行的命令写出来，便于复跑）

#### 4.3.2 backend=difssynth（实现细则）

实现策略（建议）：

* 要求用户提供 `--difsynth_repo /path/to/DiffSynth-Studio`
* `train` 内部用 `subprocess.run()` 执行 “accelerate launch …” 或执行仓库自带脚本
* Demo 只规定“你必须把 dataset_dir 注入到后端命令里”，至于后端参数名以仓库为准（codex/gemini 在实现时会去读脚本/README）

> 你在 spec 里必须写明：**train 后端命令必须包含 `--zero_cond_t`**（对 2511）([GitHub][1])

---

### 4.4 子命令四：`infer`（LoRA 推理验收：diffusers）

用途：用 base_model + lora 在 val 集上跑推理，输出结果图

**命令：**

```bash
python -m qwen_edit_demo infer \
  --dataset_dir /path/out_dataset \
  --split val \
  --base_model Qwen/Qwen-Image-Edit-2511 \
  --lora_path /path/runs/exp001/lora.safetensors \
  --out_dir /path/infer_out/exp001_val \
  --num_inference_steps 20 \
  --true_cfg_scale 4.0 \
  --negative_prompt " " \
  --height 512 \
  --width 512 \
  --seed 0
```

**推理实现硬性要求（必须写清）：**

1. `diffusers` 可能需要从源码安装（main 分支）([Hugging Face][3])
2. Pipeline 用法（示意）：

   * `pipe = QwenImageEditPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)`
   * `pipe.load_lora_weights(lora_path)`
3. 输入图像参数：

   * `QwenImageEditPipeline.__call__` 的参数名是 `image`，支持 list([Hugging Face][3])
   * 对 2511：即使单图，也用 `image=[pil_image]`（list 形式）([GitHub][2])
4. CFG：

   * 用 `true_cfg_scale` + `negative_prompt`（例如 `" "`）来启用 CFG([Hugging Face][3])

**infer 输出目录结构（建议）：**

```text
infer_out/exp001_val/
  0001/
    control.png
    target.png
    pred.png
    prompt.txt
  0002/
    ...
  index.csv                 # 汇总表：id, prompt, paths...
  contact_sheet.png         # 可选：拼图（每行 1 sample）
```

---

### 4.5 子命令五：`report`（可选但强烈建议）

用途：生成一个对人类友好的验收页面（HTML 或 Markdown）

**命令：**

```bash
python -m qwen_edit_demo report \
  --infer_dir /path/infer_out/exp001_val \
  --out_file /path/infer_out/exp001_val/report.md
```

内容建议：

* 表格：id / prompt / 原图 / 目标图 / 输出图（Markdown 里可用相对路径）
* 统计：失败样本数（推理异常 / 图像损坏）

---

## 5. 依赖与环境（实现时必须写 requirements）

### 5.1 Python 版本

* Python >= 3.10

### 5.2 最小依赖（建议）

* `torch`
* `diffusers`（建议支持从源码安装，因 QwenImage 文档为 main 版本提示）([Hugging Face][3])
* `transformers`
* `accelerate`
* `safetensors`
* `pillow`
* `tqdm`
* `typer`（或 argparse）
* `pyyaml`（如果要写 config）

---

## 6. 验收标准（必须可判定）

### 6.1 数据侧

* `inspect`：能正确报出缺失项，且 exit code 符合约定
* `prepare`：能生成：

  * `metadata.jsonl` 且每行字段齐全（id/control_image/image/prompt）
  * `images/control` 与 `images/target` 文件数量正确
  * `prompts/` 下 txt 数量正确

### 6.2 推理侧

* `infer`：对 val 集每个样本生成 `pred.png`
* 对 2511：传入 `image=[...]` 的 list 形式（单图也 list）([GitHub][2])
* CFG 参数使用 `true_cfg_scale` + `negative_prompt`（例如 `" "`）([Hugging Face][3])

### 6.3 训练侧（若实现 diffsynth 后端）

* 当 base_model=Qwen/Qwen-Image-Edit-2511：

  * 默认必须启用 `--zero_cond_t`([GitHub][1])
* `train` 运行后必须把“真实执行命令”落盘到 `cmd.sh`

---

## 7. 关键坑位提示（写进 help 文案/README）

1. **`--zero_cond_t`**：2511 LoRA 训练建议开启，Demo 默认强制为 1。([GitHub][1])
2. **multi-image 输入**：2511 单图也用 list。([GitHub][2])
3. **CFG**：用 `true_cfg_scale`，并传 `negative_prompt`（哪怕 `" "`）。([Hugging Face][3])
4. **diffusers 版本**：可能需要从 main 分支安装。([Hugging Face][3])
5. 数据里若存在同 stem 多扩展名文件（`0001.png` + `0001.jpg`），默认直接报错，避免训练集不确定性。

---

## 8. Demo README（给 codex/gemini-cli 生成代码用的“最小文案模板”）

> 你可以把下面这段直接作为生成后仓库的 `README.md` 骨架：

```md
# qwen_edit_demo

一个 Python CLI：把 (原图 A/) + (目标图+指令 B/) 转成 Qwen-Image-Edit LoRA 训练可用的数据清单，并支持 LoRA 推理验收。

## Quickstart

### 1) Inspect
python -m qwen_edit_demo inspect --src_dir A_src --tgt_dir B_tgt

### 2) Prepare
python -m qwen_edit_demo prepare --src_dir A_src --tgt_dir B_tgt --out_dir out_dataset

### 3) Train (backend 可选)
python -m qwen_edit_demo train --dataset_dir out_dataset --backend diffsynth --base_model Qwen/Qwen-Image-Edit-2511 --output_dir runs/exp001 --zero_cond_t 1

### 4) Infer
python -m qwen_edit_demo infer --dataset_dir out_dataset --split val --base_model Qwen/Qwen-Image-Edit-2511 --lora_path runs/exp001/lora.safetensors --out_dir infer_out/exp001_val
```

---

## 9. 实现建议（给 codex/gemini-cli 的工程拆分）

推荐文件结构（实现者可照抄）：

```text
qwen_edit_demo/
  __init__.py
  __main__.py          # python -m qwen_edit_demo
  cli.py               # typer/argparse
  scan.py              # 扫描配对逻辑
  prepare.py           # copy/link + metadata.jsonl
  train.py             # backend 封装（subprocess）
  infer.py             # diffusers 推理
  report.py            # 输出 markdown/html
  utils.py
```


[1]: https://github.com/ostris/ai-toolkit/issues/602 "--zero_cond_t # This is a special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model. · Issue #602 · ostris/ai-toolkit · GitHub"
[2]: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511.py?utm_source=chatgpt.com "Qwen-Image-Edit-2511.py - modelscope/DiffSynth-Studio"
[3]: https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage "QwenImage"
[4]: https://docs.oxen.ai/examples/fine-tuning/image_editing "‍ Image Editing - Oxen.ai"

