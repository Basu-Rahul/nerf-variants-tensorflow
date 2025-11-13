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
```

## 🧩 Implementation Notes
- Uses **positional encoding**, **stratified sampling**, and **volume rendering**.
- Supports configurable sample counts (`N_samples`) and MLP depths.
- Final section renders all four trained variants on a **shared test view** for visual comparison.

---

## 🧠 References

1. **TinyNeRF** – [Colab tutorial](https://github.com/yenchenlin/nerf-pytorch)  
2. **FreeNeRF** – *Yu et al., “FreeNeRF: Improving Few-shot NeRF with Free-form Radiance Field Regularization,” ICCV 2023.*  
3. **SimpleNeRF** – *Somraj et al., “SimpleNeRF: Lightweight NeRF via Simplified MLP and Positional Encoding,” 2023.*  
4. **FrugalNeRF** – *Lin et al., “FrugalNeRF: Parameter-efficient Neural Radiance Fields,” CVPR 2025.*

---

## 📜 License
This project is released under the **MIT License**.  
Feel free to use, modify, and cite appropriately.

---

## 👤 Author
Rahul Basu  
📧 rahulbasutigps@gmail.com  


