import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

def make_dataset(global_batch_size):
    """
    Returns (x, x) pairs for unsupervised VAE training,
    loading MNIST from the local 'data/' folder.
    """
    data_path = Path(__file__).parent.parent / "data"
    def _prep(sample):
        x = tf.cast(sample["image"], tf.float32) / 255.0
        return x, x

    return (
        tfds.load(
            "mnist",
            split="train",
            shuffle_files=True,
            data_dir=str(data_path)
        )
        .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(60_000)
        .batch(global_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
