# Image Denoising Filters Comparison  
A practical evaluation of three classical image denoising techniques — Gaussian smoothing, Median filtering, and a Combined Smoothing→Median approach — applied to the standard *Lena* image corrupted with both salt-and-pepper and Gaussian noise.

This project demonstrates noise modeling, classical image filtering, and quantitative evaluation using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index), which are widely used in computer vision and image-processing pipelines.

---

## 📌 Project Overview

In many real-world edge AI and computer vision applications, images are corrupted by different types of noise. Each noise type requires the appropriate filtering method.

We simulate a noisy version of the *Lena* image using:

- **Salt-and-pepper noise** (impulse noise: random white/black pixels)  
- **Gaussian noise** (continuous intensity variation)

We then apply three filtering methods:

### 🔹 1. Gaussian Smoothing (Smoothing-only)
- Reduces Gaussian noise well  
- Cannot fully remove impulse noise  
- Tends to blur edges  

### 🔹 2. Median Filtering (Median-only)
- Excellent for impulse noise  
- Preserves edges better  
- Does not reduce Gaussian noise as effectively  

### 🔹 3. Combined Filter (Gaussian Smoothing → Median Filtering)
- First reduces Gaussian noise  
- Then removes remaining impulse noise  
- Often provides the best balance depending on noise strength  

---

## 🧪 Evaluation Metrics
We compute:

### 📌 PSNR — Peak Signal-to-Noise Ratio  
Measures pixel-level reconstruction quality.

### 📌 SSIM — Structural Similarity Index  
Measures perceptual and structural similarity.

These metrics help determine which filter performs best under mixed noise conditions.

---

## 📂 Folder Structure
