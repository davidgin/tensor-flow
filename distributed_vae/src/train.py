import os, json, datetime, pathlib, socket
import tensorflow as tf
from data import make_dataset
from model import VAE

# Config
BATCH_PER_REPLICA = 128
LATENT_DIM        = 48
EPOCHS            = 10

# Multi‑worker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
task     = json.loads(os.getenv("TF_CONFIG", "{}")).get("task", {})
print(f"[{socket.gethostname()}] Task={task} | Replicas={strategy.num_replicas_in_sync}")

# Dataset
with strategy.scope():
    global_batch = BATCH_PER_REPLICA * strategy.num_replicas_in_sync
    dataset      = make_dataset(global_batch)

# Model
with strategy.scope():
    model = VAE(LATENT_DIM)
    model.compile(optimizer=tf.keras.optimizers.Adam())

# Logging
log_dir = pathlib.Path("runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer  = tf.summary.create_file_writer(str(log_dir))

# Training loop
for epoch in range(EPOCHS):
    for step, batch in enumerate(dataset):
        logs = model.train_on_batch(batch, return_dict=True)
        if step % 100 == 0 and task.get("index", 0) == 0:
            print(f"Epoch {epoch+1} Step {step:04d} loss={logs['loss']:.4f}")
            with writer.as_default():
                tf.summary.scalar("loss", logs["loss"], step=epoch*500 + step)

# Save on worker‑0
if task.get("index", 0) == 0:
    out = pathlib.Path("exported_vae")
    out.mkdir(exist_ok=True)
    model.save(out, include_optimizer=False)
    print("✓ Model saved to", out)
