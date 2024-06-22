# src/train_vae.py
# Unsupervised Variational Auto‑Encoder with MultiWorkerMirroredStrategy
import os, json, datetime, pathlib, socket, tensorflow as tf, tensorflow_datasets as tfds

strategy = tf.distribute.MultiWorkerMirroredStrategy()
task     = json.loads(os.getenv("TF_CONFIG", "{}")).get("task", {})
print(f"[{socket.gethostname()}] Task={task}  Replicas={strategy.num_replicas_in_sync}")

GLOBAL_BATCH = 128 * strategy.num_replicas_in_sync
LATENT_DIM   = 48
EPOCHS       = 10
LOG_DIR      = pathlib.Path("runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# dataset
def make_ds():
    def _prep(s):
        x = tf.cast(s["image"], tf.float32) / 255.0
        return x, x
    return (tfds.load("mnist", split="train", shuffle_files=True)
              .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
              .cache()
              .shuffle(60_000)
              .batch(GLOBAL_BATCH, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))

# model
class Encoder(tf.keras.layers.Layer):
    def __init__(self, z): super().__init__(); self.net=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,2,activation="relu"),
        tf.keras.layers.Conv2D(64,3,2,activation="relu"),
        tf.keras.layers.Flatten()]); self.mu=tf.keras.layers.Dense(z); self.logv=tf.keras.layers.Dense(z)
    def call(self,x): x=self.net(x); return self.mu(x),self.logv(x)

class Decoder(tf.keras.layers.Layer):
    def __init__(self,z): super().__init__(); self.net=tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*64,activation="relu"),
        tf.keras.layers.Reshape((7,7,64)),
        tf.keras.layers.Conv2DTranspose(64,3,2,"same",activation="relu"),
        tf.keras.layers.Conv2DTranspose(32,3,2,"same",activation="relu"),
        tf.keras.layers.Conv2DTranspose(1,3,padding="same",activation="sigmoid")])
    def call(self,z): return self.net(z)

class VAE(tf.keras.Model):
    def __init__(self,z): super().__init__(); self.enc,self.dec=Encoder(z),Decoder(z); self.mse=tf.keras.losses.MeanSquaredError(reduction='none')
    def train_step(self,data):
        x,_=data
        with tf.GradientTape() as tape:
            mu,logv=self.enc(x); eps=tf.random.normal(tf.shape(mu)); z=mu+tf.exp(0.5*logv)*eps; xh=self.dec(z)
            recon=tf.reduce_sum(self.mse(x,xh),[1,2,3]); kl=-0.5*tf.reduce_sum(1+logv-tf.square(mu)-tf.exp(logv),1)
            loss=tf.reduce_mean(recon+kl)
        grads=tape.gradient(loss,self.trainable_variables); self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {"loss":loss}
    def sample(self,n=100): return self.dec(tf.random.normal((n,LATENT_DIM)))

with strategy.scope():
    ds=make_ds(); model=VAE(LATENT_DIM); model.compile(optimizer=tf.keras.optimizers.Adam())

writer=tf.summary.create_file_writer(str(LOG_DIR))
for epoch in range(EPOCHS):
    for step,b in enumerate(ds):
        logs=model.train_on_batch(b,return_dict=True)
        if step%100==0 and task.get("index",0)==0:
            print(f"Epoch {epoch+1} Step {step:04d} loss={logs['loss']:.4f}")
            with writer.as_default(): tf.summary.scalar("loss",logs["loss"],step=epoch*500+step)

if task.get("index",0)==0:
    export=pathlib.Path("exported_vae"); export.mkdir(exist_ok=True)
    model.save(export,include_optimizer=False); print("✓ Saved model to",export)
# Refactor encoder to GELU (2024-07-29T06:45:27)
# Add TensorBoard logging (2023-05-07T18:11:18)
# Pin tfds version (2023-11-06T06:30:49)
# Expose LATENT_DIM via env (2023-09-29T10:39:07)
# Improve launch script GPU mask (2024-10-08T02:58:32)
# Track recon & KL separately (2024-08-25T11:46:30)
# CLI flag for batch size (2025-01-29T18:59:09)
# Switch recon loss BCE→MSE (2023-06-21T23:10:06)
# README: multi‑host guidance (2024-02-21T12:04:13)
# .gitignore TensorBoard logs (2023-05-11T06:02:50)
# Add GPU Dockerfile (2023-09-26T02:32:14)
# Early‑stopping callback (2024-04-22T09:49:28)
# Unit test encoder output (2023-05-08T20:54:36)
# Black formatter pre‑commit (2023-09-11T15:38:08)
# Optimize shuffle buffer (2024-08-05T21:58:31)
# Enable mixed‑precision (2024-05-21T07:22:29)
# Ignore sample_grid.png (2023-09-27T10:07:10)
# Explain TF_CONFIG anatomy (2024-06-22T15:56:05)
