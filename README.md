# Distributed Unsupervised VAE (TensorFlow 2.x)

This example demonstrates **unsupervised generative training** of a Variational Auto‑Encoder (VAE) on the MNIST dataset using TensorFlow’s `tf.distribute.MultiWorkerMirroredStrategy`. The code is modularized into separate components for data loading, model definition, and training logic, and supports running across multiple workers (e.g., on a multi‑host cluster or locally via Docker or shell scripts).

## 📁 Project Structure

```
distributed_vae/
├── data/                         <-- TFDS local cache (empty initially)
├── launch_multiworker.sh        <-- Script to launch 2 workers locally
├── README.md                    <-- This file
├── requirements.txt             <-- Python dependencies
└── src/
    ├── __init__.py              <-- Python package marker
    ├── data.py                  <-- Dataset loader (TFDS, local `data/`)
    ├── model.py                 <-- Encoder, Decoder, VAE definitions
    └── train.py                 <-- Training loop + distribution logic
```

## 🚀 Quick Start

1. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **(Optional) Pre‑download MNIST into `data/`**

   Ensures offline reproducibility and removes TFDS download latency:

   ```bash
   python - << 'EOF'
import tensorflow_datasets as tfds

tfds.load(
    "mnist",
    split=["train", "test"],
    data_dir="data",
    download=True
)
print("✅ MNIST downloaded into data/")
EOF
   ```

4. **Launch the two‑worker demo**

   ```bash
   chmod +x launch_multiworker.sh
   ./launch_multiworker.sh
   ```

   - Worker 0 binds to port 12345 and Worker 1 to port 12346.
   - Training logs (loss) will appear in the console every 100 steps.
   - Upon completion, Worker 0 writes a SavedModel to `exported_vae/`.

5. **Inspect results**

   - **Model export:** `distributed_vae/exported_vae/` contains the SavedModel artifacts.
   - **TensorBoard logs:** under `distributed_vae/runs/<timestamp>/`.
     ```bash
     tensorboard --logdir runs
     ```

## ⚙️ Running on a Multi‑Host Cluster

1. Copy the **entire** `distributed_vae/` folder to each node (or mount via NFS).
2. Edit `launch_multiworker.sh`, replacing `127.0.0.1` with each host’s IP and ensure the ports are open.
3. Run the same script on **all** machines:
   ```bash
   ./launch_multiworker.sh
   ```

TensorFlow will:
- Automatically shard the dataset across replicas.
- Perform synchronous gradient aggregation.
- Recover from worker restarts if configured.

## 🛠 Customization

- **Swap in a new dataset**: Modify `src/data.py` to load your own images (return `(x, x)` pairs).
- **Expand the VAE**: Implement more complex encoders/decoders or integrate diffusion/backbone models in `src/model.py`.
- **Adjust hyperparameters**: Change batch size, latent dimension, or number of epochs in `src/train.py`.
- **Use ParameterServerStrategy**: For very large models (>20 GB), switch distribution strategy.

## 🤝 Contributing

Feel free to open issues or submit pull requests to:
- Add more examples (e.g., autoencoders for different datasets).
- Integrate logging frameworks or metrics (e.g., WandB, MLflow).
- Improve error handling and CLI args parsing.

---

© 2025 Your Name or Organization

