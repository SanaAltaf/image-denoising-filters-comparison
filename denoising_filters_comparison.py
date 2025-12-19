# ===============================================================
# Homework: Denoising Lena with S&P + Gaussian noise
# Methods: (1) Smoothing-only, (2) Median-only, (3) Smoothing→Median
# Libraries: numpy, opencv-python (cv2), scikit-image, matplotlib
# ===============================================================

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from skimage import img_as_float, data
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Helper: robust image loading
# -----------------------------
def load_grayscale_image(local_paths):
    """
    Try multiple possible local paths for Lena. If none exist, fall back to a
    known skimage sample (astronaut), converted to grayscale.

    Args:
        local_paths (list[Path]): candidate absolute/relative paths.

    Returns:
        img (float32): HxW image in [0,1]
        used_path (Path or str): the path used, or 'skimage_fallback'
    """
    for p in local_paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return img_as_float(arr).astype(np.float32), p
    # Fallback: use skimage astronaut (green channel) as grayscale
    arr = img_as_float(data.astronaut())[:, :, 1].astype(np.float32)
    return arr, "skimage_fallback"


# ===============================================================
# STEP 1) Load Lena (edit LOCAL_LENA_* to your actual file if needed)
# ===============================================================
# TIP: if you already confirmed your path earlier, put it in LOCAL_LENA_PATH.
# Leave others as None. The loader will try each in order.

LOCAL_LENA_PATH = Path("/Users/sana/Desktop/FA25Courses/Comp vision with AI/Projects/Homework1/lena.png")  # <-- update if needed
ALT_RELATIVE    = Path("lena.png")       # try current folder as well
ALT_JPG        = Path("lena.jpg")        # try common extension
CANDIDATES = [LOCAL_LENA_PATH, ALT_RELATIVE, ALT_JPG]

img, used = load_grayscale_image(CANDIDATES)
print(f"[INFO] Loaded image from: {used}")

# ===============================================================
# STEP 2) Add Noise: Salt & Pepper + Gaussian
# ===============================================================
# Reproducibility: set a random seed so results match each run
np.random.seed(42)

# Choose noise levels (you can tweak to explore):
sp_amount = 0.05   # proportion of pixels replaced by salt/pepper (typical 0.02–0.10)
sigma     = 0.05   # Gaussian std dev in [0,1] range (typical 0.02–0.10)

# Apply noises
noisy_sp   = random_noise(img, mode='s&p', amount=sp_amount)          # S&P first
noisy_both = random_noise(noisy_sp, mode='gaussian', var=sigma**2)    # then Gaussian

# ===============================================================
# STEP 3) Denoising Methods
# ===============================================================
# Important: OpenCV's medianBlur expects uint8. We'll convert to/from [0,1].

def to_uint8(x):
    return np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)

def to_float01(x):
    return x.astype(np.float32) / 255.0

# 3.1) Smoothing-only (Gaussian blur): good for Gaussian noise; leaves S&P specks
smoothed = cv2.GaussianBlur(noisy_both, ksize=(5, 5), sigmaX=1)

# 3.2) Median-only: great for S&P; less effective on pure Gaussian
median_u8 = cv2.medianBlur(to_uint8(noisy_both), ksize=5)
median = to_float01(median_u8)

# 3.3) Combined (Smoothing → Median): reduce Gaussian first, then remove impulses
smooth_then_median_u8 = cv2.medianBlur(to_uint8(smoothed), ksize=5)
smooth_then_median = to_float01(smooth_then_median_u8)

# ===============================================================
# STEP 4) Quantitative Evaluation (PSNR, SSIM) vs. CLEAN image
# ===============================================================
results = {
    "Noisy Input": (
        psnr(img, noisy_both, data_range=1.0),
        ssim(img, noisy_both, data_range=1.0),
    ),
    "Smoothing-only": (
        psnr(img, smoothed, data_range=1.0),
        ssim(img, smoothed, data_range=1.0),
    ),
    "Median-only": (
        psnr(img, median, data_range=1.0),
        ssim(img, median, data_range=1.0),
    ),
    "Smoothing → Median": (
        psnr(img, smooth_then_median, data_range=1.0),
        ssim(img, smooth_then_median, data_range=1.0),
    ),
}

print("\n=== Denoising Results (vs. clean Lena) ===")
for k, (p, s) in results.items():
    print(f"{k:18s} | PSNR = {p:6.2f} dB | SSIM = {s:0.4f}")

# ===============================================================
# STEP 5) Visualization
# ===============================================================
# Show original + noisy variants (top), and the three methods (bottom).
# ===============================================================
# Visualization: include all methods (2 rows × 4 columns)
# ===============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

# Row 1: original + noisy images
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Lena"); axes[0].axis('off')

axes[1].imshow(noisy_sp, cmap='gray')
axes[1].set_title(f"Salt & Pepper (amount={sp_amount})"); axes[1].axis('off')

axes[2].imshow(noisy_both, cmap='gray')
axes[2].set_title(f"S&P + Gaussian (σ={sigma})"); axes[2].axis('off')

axes[3].imshow(noisy_both, cmap='gray')
axes[3].set_title("Noisy Input"); axes[3].axis('off')

# Row 2: filtered results
axes[4].imshow(smoothed, cmap='gray')
axes[4].set_title("Smoothing-only"); axes[4].axis('off')

axes[5].imshow(median, cmap='gray')
axes[5].set_title("Median-only"); axes[5].axis('off')

axes[6].imshow(smooth_then_median, cmap='gray')
axes[6].set_title("Smoothing → Median"); axes[6].axis('off')

# leave the last slot empty (or you can put something like the clean image again)
axes[7].axis('off')

plt.tight_layout()
plt.show()

# ===============================================================
# STEP 6) Save outputs (useful for submission/report)
# ===============================================================
outdir = Path("./outputs")
outdir.mkdir(exist_ok=True)

def save_gray(name, arr01):
    cv2.imwrite(str(outdir / name), to_uint8(arr01))

save_gray("01_original.png", img)
save_gray("02_noisy_sp.png", noisy_sp)
save_gray("03_noisy_both.png", noisy_both)
save_gray("11_smoothing_only.png", smoothed)
save_gray("12_median_only.png", median)
save_gray("13_smoothing_then_median.png", smooth_then_median)

print(f"[INFO] Saved images to: {outdir.resolve()}")
