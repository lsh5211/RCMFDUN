
# HTDIDUN (Image Compressed Sensing Reconstruction)

This repository contains the official implementation of **HTDIDUN** for image compressed sensing (CS) reconstruction.

 **Pretrained weights (Google Drive):**
  [https://drive.google.com/drive/folders/1tiFbdMkyR3N7B9x4EgtcHWXChJkA2D1m?usp=drive\_link](https://drive.google.com/drive/folders/1tiFbdMkyR3N7B9x4EgtcHWXChJkA2D1m?usp=drive_link)

---

## Table of Contents

* [Environment](#environment)
* [Project Structure](#project-structure)
* [Data Preparation](#data-preparation)
* [Pretrained Weights](#pretrained-weights)
* [Quick Start (Test Only)](#quick-start-test-only)
* [Command Examples](#command-examples)
* [Results](#results)
* [Training (Optional)](#training-optional)
* [Citation](#citation)
* [License](#license)

---

## Environment

* Python ≥ 3.8
* PyTorch ≥ 1.10, torchvision (CUDA optional)
* Other: `numpy`, `scikit-image`, `opencv-python`, `tqdm`

Install (example):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-image opencv-python tqdm
```

---

## Project Structure

```
HTDIDUN-main/
├─ data/              # place test sets here (e.g., Set11)
├─ figs/
├─ log/
├─ model/             # place pretrained .pth/.pkl here
├─ test_out/          # outputs will be saved here
├─ MRI.py
├─ pinpufenxi.py
├─ RCMFDUN.py
├─ test.py
├─ train.py
└─ utils.py
```

---

## Data Preparation

1. **Create a test set folder** under `./data`.
   Recommended layout:

   ```
   data/
   └─ Set11/
      ├─ 01.png
      ├─ 02.png
      └─ ...
   ```

   * Images can be PNG/JPG (grayscale or RGB).
   * If your `test.py` expects a different name, keep the folder name consistent with the `--testset_name`/`--testset` flag you use below.

---

## Pretrained Weights

1. Download the weights from Google Drive:
   [https://drive.google.com/drive/folders/1tiFbdMkyR3N7B9x4EgtcHWXChJkA2D1m?usp=drive\_link](https://drive.google.com/drive/folders/1tiFbdMkyR3N7B9x4EgtcHWXChJkA2D1m?usp=drive_link)

2. **Put the downloaded files into `./model/`.**
   Example:

   ```
   model/
   ├─ mark10_32_ratio_0.10_layer_10_block_32.pth
   ├─ mark10_32_ratio_0.25_layer_10_block_32.pth
   └─ mark10_32_ratio_0.50_layer_10_block_32.pth
   ```

---

## Quick Start (Test Only)

With the test set prepared in `./data/Set11` and weights placed in `./model/`, run:

```bash
python test.py \
  --weights ./model/mark10_32_ratio_0.25_layer_10_block_32.pth \
  --testset_name Set11 \
  --save_dir ./test_out
```

* `--weights`: path to the chosen pretrained model.
* `--testset_name`: folder name under `./data/` containing test images.
* `--save_dir`: where to save reconstructions and logs.

> If your script uses different flag names (e.g., `--model_path`, `--testset`), just replace accordingly. The three key ideas remain the same:
>
> 1. **test images** go to `./data/<your_testset>`
> 2. **pretrained weights** go to `./model/`
> 3. outputs are written to `./test_out/`

---

## Command Examples

**1) Test with 10% sampling ratio model**

```bash
python test.py \
  --weights ./model/mark10_32_ratio_0.10_layer_10_block_32.pth \
  --testset_name Set11 \
  --save_dir ./test_out/ratio10
```

**2) Test with 25% sampling ratio model**

```bash
python test.py \
  --weights ./model/mark10_32_ratio_0.25_layer_10_block_32.pth \
  --testset_name Set11 \
  --save_dir ./test_out/ratio25
```

**3) Test with 50% sampling ratio model**

```bash
python test.py \
  --weights ./model/mark10_32_ratio_0.50_layer_10_block_32.pth \
  --testset_name Set11 \
  --save_dir ./test_out/ratio50
```

Optional common flags (use if they exist in your `argparse`):

```bash
--block_size 32 --phase_num 10 --gpu_list 0
```

---

## Results

After running, reconstructed images/metrics will appear in `./test_out/...`.
You can customize the path via `--save_dir`.

---

## Training (Optional)

If you want to train from scratch (weights will be written to `./model/`):

```bash
python train.py \
  --phase_num 10 \
  --block_size 32 \
  --end_epoch 600 \
  --data_dir ./data \
  --log_dir ./log \
  --model_dir ./model
```



