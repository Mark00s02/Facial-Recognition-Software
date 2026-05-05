"""
Downloads all model files needed by the system.
Run once:  python download_models.py

  DL face recognition (YuNet + SFace ONNX) — enables 80-95 % confidence scores
  Age / Gender Caffe models                 — enables the overlay engine
"""

import os
import sys
import urllib.request
import urllib.error

MODELS_DIR = "models"

# ── Prototxt content (embedded — no download needed) ─────────────────────────

DEPLOY_AGE = """\
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227
layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1"
  convolution_param { num_output: 96 kernel_size: 7 stride: 4 } }
layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }
layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "norm1" type: "LRN" bottom: "pool1" top: "norm1"
  lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 } }
layer { name: "conv2" type: "Convolution" bottom: "norm1" top: "conv2"
  convolution_param { num_output: 256 pad: 2 kernel_size: 5 } }
layer { name: "relu2" type: "ReLU" bottom: "conv2" top: "conv2" }
layer { name: "pool2" type: "Pooling" bottom: "conv2" top: "pool2"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "conv3" type: "Convolution" bottom: "pool2" top: "conv3"
  convolution_param { num_output: 384 pad: 1 kernel_size: 3 } }
layer { name: "relu3" type: "ReLU" bottom: "conv3" top: "conv3" }
layer { name: "pool5" type: "Pooling" bottom: "conv3" top: "pool5"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "fc6" type: "InnerProduct" bottom: "pool5" top: "fc6"
  inner_product_param { num_output: 512 } }
layer { name: "relu6" type: "ReLU" bottom: "fc6" top: "fc6" }
layer { name: "drop6" type: "Dropout" bottom: "fc6" top: "fc6"
  dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc7" type: "InnerProduct" bottom: "fc6" top: "fc7"
  inner_product_param { num_output: 512 } }
layer { name: "relu7" type: "ReLU" bottom: "fc7" top: "fc7" }
layer { name: "drop7" type: "Dropout" bottom: "fc7" top: "fc7"
  dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc8" type: "InnerProduct" bottom: "fc7" top: "fc8"
  inner_product_param { num_output: 8 } }
layer { name: "prob" type: "Softmax" bottom: "fc8" top: "prob" }
"""

DEPLOY_GENDER = """\
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227
layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1"
  convolution_param { num_output: 96 kernel_size: 7 stride: 4 } }
layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }
layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "norm1" type: "LRN" bottom: "pool1" top: "norm1"
  lrn_param { local_size: 5 alpha: 0.0001 beta: 0.75 } }
layer { name: "conv2" type: "Convolution" bottom: "norm1" top: "conv2"
  convolution_param { num_output: 256 pad: 2 kernel_size: 5 } }
layer { name: "relu2" type: "ReLU" bottom: "conv2" top: "conv2" }
layer { name: "pool2" type: "Pooling" bottom: "conv2" top: "pool2"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "conv3" type: "Convolution" bottom: "pool2" top: "conv3"
  convolution_param { num_output: 384 pad: 1 kernel_size: 3 } }
layer { name: "relu3" type: "ReLU" bottom: "conv3" top: "conv3" }
layer { name: "pool5" type: "Pooling" bottom: "conv3" top: "pool5"
  pooling_param { pool: MAX kernel_size: 3 stride: 2 } }
layer { name: "fc6" type: "InnerProduct" bottom: "pool5" top: "fc6"
  inner_product_param { num_output: 512 } }
layer { name: "relu6" type: "ReLU" bottom: "fc6" top: "fc6" }
layer { name: "drop6" type: "Dropout" bottom: "fc6" top: "fc6"
  dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc7" type: "InnerProduct" bottom: "fc6" top: "fc7"
  inner_product_param { num_output: 512 } }
layer { name: "relu7" type: "ReLU" bottom: "fc7" top: "fc7" }
layer { name: "drop7" type: "Dropout" bottom: "fc7" top: "fc7"
  dropout_param { dropout_ratio: 0.5 } }
layer { name: "fc8" type: "InnerProduct" bottom: "fc7" top: "fc8"
  inner_product_param { num_output: 2 } }
layer { name: "prob" type: "Softmax" bottom: "fc8" top: "prob" }
"""

# ── Caffemodel mirrors (tried in order) ───────────────────────────────────────
# Each entry is a list of URLs tried top-to-bottom until one succeeds.

CAFFEMODEL_SOURCES = {
    "age_net.caffemodel": [
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/age_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models_caffe/age_net.caffemodel",
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
    ],
    "gender_net.caffemodel": [
        "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel",
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models_caffe/gender_net.caffemodel",
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
    ],
}

MIN_MODEL_BYTES    = 50 * 1024 * 1024   # a real caffemodel is ~98 MB; reject tiny files
MIN_ONNX_BYTES_DET = 300 * 1024         # YuNet  ~400 KB
MIN_ONNX_BYTES_REC = 30  * 1024 * 1024 # SFace  ~37 MB

# ── DL (YuNet + SFace) ONNX model sources ────────────────────────────────────
ONNX_SOURCES = {
    "face_detection_yunet_2023mar.onnx": {
        "min_bytes": MIN_ONNX_BYTES_DET,
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://github.com/ShiqiYu/libfacedetection.train/raw/master/tasks/task1/onnx/yunet.onnx",
        ],
    },
    "face_recognition_sface_2021dec.onnx": {
        "min_bytes": MIN_ONNX_BYTES_REC,
        "urls": [
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 4) + "-" * (25 - int(pct / 4))
        mb  = downloaded / 1_048_576
        print(f"\r    [{bar}] {pct:5.1f}%  {mb:.1f} MB   ", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r    {mb:.1f} MB downloaded…", end="", flush=True)


def _try_download(url: str, dest: str, min_bytes: int) -> bool:
    """Download url → dest. Returns True on success."""
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()
        size = os.path.getsize(tmp)
        if size < min_bytes:
            print(f"    ✗  File too small ({size} bytes) — likely a redirect page, skipping.")
            os.remove(tmp)
            return False
        os.replace(tmp, dest)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"\n    ✗  {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_ok = True

    # ── DL face recognition models (YuNet + SFace ONNX) ──────────────────────
    print("── Deep Learning Face Recognition (YuNet + SFace) ──")
    for filename, info in ONNX_SOURCES.items():
        dest      = os.path.join(MODELS_DIR, filename)
        min_bytes = info["min_bytes"]
        if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
            mb = os.path.getsize(dest) / 1_048_576
            print(f"  ✓  {filename} already exists ({mb:.1f} MB).")
            continue

        size_hint = "~0.4 MB" if "yunet" in filename else "~37 MB"
        print(f"  Downloading  {filename}  ({size_hint}) …")
        success = False
        for i, url in enumerate(info["urls"], 1):
            print(f"    Mirror {i}/{len(info['urls'])}: {url[:80]}…")
            if _try_download(url, dest, min_bytes):
                mb = os.path.getsize(dest) / 1_048_576
                print(f"  ✓  Saved ({mb:.1f} MB)")
                success = True
                break

        if not success:
            print(f"\n  ✗  Could not download {filename}.")
            print(f"     Manually place the file in:  {os.path.abspath(MODELS_DIR)}/")
            print(f"     Source: https://github.com/opencv/opencv_zoo")
            all_ok = False
        print()

    # ── Age / Gender Caffe prototxt files (embedded) ──────────────────────────
    print("── Age / Gender Overlay Models ──")
    for filename, content in [
        ("deploy_age.prototxt",    DEPLOY_AGE),
        ("deploy_gender.prototxt", DEPLOY_GENDER),
    ]:
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  ✓  {filename} already exists.")
        else:
            with open(dest, "w") as f:
                f.write(content)
            print(f"  ✓  {filename} written.")

    print()

    # ── Age / Gender Caffe binary weights ─────────────────────────────────────
    for filename, mirrors in CAFFEMODEL_SOURCES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) >= MIN_MODEL_BYTES:
            print(f"  ✓  {filename} already exists ({os.path.getsize(dest)//1_048_576} MB).")
            continue

        print(f"  Downloading  {filename}  (~98 MB) …")
        success = False
        for i, url in enumerate(mirrors, 1):
            print(f"    Mirror {i}/{len(mirrors)}: {url[:72]}…")
            if _try_download(url, dest, MIN_MODEL_BYTES):
                mb = os.path.getsize(dest) // 1_048_576
                print(f"  ✓  Saved ({mb} MB)")
                success = True
                break

        if not success:
            print(f"\n  ✗  All mirrors failed for {filename}.")
            print("     Manual download: search 'age_net.caffemodel download github'")
            print(f"     Place the file in:  {os.path.abspath(MODELS_DIR)}/")
            all_ok = False
        print()

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  Facial Recognition System — Model Downloader")
    print("=" * 60)
    ok = download()
    print("=" * 60)
    if ok:
        print("  All models ready.")
        print("  ◈ DL mode (YuNet + SFace) — 80-95 % confidence")
        print("  ◈ Age / Gender overlay enabled")
    else:
        print("  Some files are missing — check messages above.")
    print("=" * 60)
    input("\nPress Enter to close…")
