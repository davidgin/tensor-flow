import tensorflow as tf
from distributed_vae.src.data import make_dataset

def test_dataset_shape_and_types(tmp_path):
    # Use a tiny “fake” TFDS dataset for speed
    ds = make_dataset(global_batch_size=16)
    elem = next(iter(ds.take(1)))
    x_batch, y_batch = elem

    # Expect 4‑D image batches and identical x==y pairs
    assert isinstance(x_batch, tf.Tensor)
    assert isinstance(y_batch, tf.Tensor)
    assert x_batch.shape[0] == 16
    assert x_batch.shape[1:] == (28, 28, 1)
    tf.debugging.assert_near(x_batch, y_batch)  # unsupervised target==input

