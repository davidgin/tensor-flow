
import tensorflow as tf
import numpy as np
from distributed_vae.src.model import Encoder, Decoder, VAE

def make_dummy_images(batch, h=28, w=28, c=1):
    return tf.constant(np.random.rand(batch, h, w, c), dtype=tf.float32)

def test_encoder_outputs_shape():
    enc = Encoder(latent_dim=12)
    x = make_dummy_images(8)
    mu, logv = enc(x)
    assert mu.shape == (8, 12)
    assert logv.shape == (8, 12)

def test_decoder_reconstruction_shape():
    dec = Decoder(latent_dim=12)
    z = tf.random.normal((8, 12))
    xh = dec(z)
    assert xh.shape == (8, 28, 28, 1)

def test_vae_train_step_and_sample():
    vae = VAE(latent_dim=12)
    # Build variables by calling once
    _ = vae(make_dummy_images(4))
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    logs = vae.train_step((make_dummy_images(4), make_dummy_images(4)))
    assert "loss" in logs
    assert logs["loss"] >= 0.0

    # Sampling should give correct shape
    samples = vae.sample(num_samples=10, latent_dim=12)
    assert samples.shape == (10, 28, 28, 1)

