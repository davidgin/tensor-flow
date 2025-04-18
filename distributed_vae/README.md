# Modular Distributed VAE (TensorFlow 2)

This example now includes a local `data/` folder for caching TFDS.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd distributed_vae
./launch_multiworker.sh
```

**Pre‑download** MNIST into `data/` (optional):
```bash
python -c "import tensorflow_datasets as tfds; tfds.load('mnist', data_dir='distributed_vae/data')"
```

Worker 0 writes `exported_vae/` when done.
