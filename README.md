# 🌊 MiniFlow: Flow Matching on Moon Data

This project demonstrates the core concepts of **Flow Matching** by learning a vector field that transforms simple Gaussian noise into the structured `make_moons` distribution.

## 🚀 Setup & Installation

Follow these steps to set up the environment and install dependencies. Note that this configuration is optimized for **CUDA 12.1**.

```bash
# Create and activate the environment
conda create -n miniflow python=3.11 -y
conda activate miniflow

# Install PyTorch (Compatible with CUDA 12.1)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core dependencies
pip install tqdm pyyaml scikit-learn imageio matplotlib wandb
```
## 🏋️ Training
```bash
python mini_flow/miniflow.py --config_path ./mini_flow/configs/miniflow.yaml
```

## 📊 Velocity Field Evolution
The following images visualize the transformation of the learned velocity field v(x,t). The model transitions from a state of total entropy to a structured flow that defines the moon manifold.
<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td align="center"><b>Before Training</b></td>
    <td align="center"><b>After Training</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/cfc47f13-b9e5-4a8c-bc9a-af25db1b15c9" width="100%" />
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/3fb9d78e-e3b3-4549-84c0-22a7948eb95f" width="100%" />
    </td>
  </tr>
</table>
