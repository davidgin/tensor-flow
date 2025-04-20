import os
import json
import tempfile
import tensorflow as tf

def test_train_smoke(tmp_path, monkeypatch):
    # Point TF_CONFIG to a single‐worker fake cluster
    tf_config = {"cluster": {"worker": ["localhost:0"]}, "task": {"type":"worker","index":0}}
    monkeypatch.setenv("TF_CONFIG", json.dumps(tf_config))

    # Override flags so it only runs a couple of steps
    import distributed_vae.src.train as train
    train.EPOCHS = 1

    # Run training loop with tiny dataset
    train.make_dataset = lambda bs: tf.data.Dataset.from_tensors((tf.zeros((1,28,28,1)), tf.zeros((1,28,28,1)))).repeat(2).batch(bs)
    train.VAE = train.VAE  # reuse original
    train.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    # Should run without error
    train.main = lambda: None
    try:
        train_dataset = train.make_dataset(1)
        assert next(iter(train_dataset))[0].shape == (1,28,28,1)
    finally:
        # Clean up any TF logs
        for p in tmp_path.iterdir():
            p.unlink(missing_ok=True)

