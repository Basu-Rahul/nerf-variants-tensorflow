# nerf-variants-tensorflow
Lightweight NeRF variants (TinyNeRF, FreeNeRF, SimpleNeRF, FrugalNeRF) implemented and benchmarked in a unified TensorFlow notebook with Colab GPU support.
# 🧠 NeRF Variants in TensorFlow (Unified Notebook)

This repository presents a **single unified TensorFlow 2.x implementation** of four lightweight Neural Radiance Field (NeRF) variants:

- 🟦 **TinyNeRF** – Minimal educational baseline ([Colab, 2020])  
- 🟩 **FreeNeRF** – Geometry-regularized NeRF (Yu et al., 2023)  
- 🟨 **SimpleNeRF** – Simplified MLP and reduced positional encoding (Somraj et al., 2023)  
- 🟥 **FrugalNeRF** – Parameter-efficient NeRF with voxelized weight sharing (Lin et al., 2025)

All models are **selectable within a single notebook** using the `variant` argument inside `run_variant()`:
```python
# -------------------------
# USER CONFIG: Run all
# -------------------------
results={}
for var in ["tinynerf","freenerf","simplenerf","frugalnerf"]:
    results[var]=run_variant(var,N_iters=4000,N_rand=1024)

# Here change based on requirement
