# TODO

- Resume the 32-card SOFY `svhn_quant` sweep after Qwen Image Edit work is done.
- Pending / interrupted `svhn_quant` tasks on the 32-card cluster:
  - `tasks 8-11`: completed before the Qwen work took over the cluster
  - `tasks 12-30`: canceled to free the 32-card cluster, including the `1-bit` and `fp32_last` control runs
- When resuming `svhn_quant`, restart from the current sweep root and re-check TensorBoard on `127.0.0.1:6008`.

- After reproducing A/B datapairs, work on `~/autodl-tmp/samples10k_lab_mel_png` -> `~/autodl-tmp/samples10k_mel_png` (transform former to latter).
- If results are bad and alignment is needed (DTW mel spectra), check `~/autodl-tmp/mel2mel_demo/`.
