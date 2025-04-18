# Distributed Unsupervised VAE (TensorFlow 2)

Train a Variational Auto‑Encoder on MNIST using
`tf.distribute.MultiWorkerMirroredStrategy`.

## Local smoke‑test
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./launch_multiworker.sh
```

Worker‑0 writes `exported_vae/` when training ends.
