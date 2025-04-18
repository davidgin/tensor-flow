#!/usr/bin/env python3
"""
create_distributed_vae_project.py
---------------------------------
Scaffolds a modular distributed‑VAE TensorFlow example, now including
a local `data/` folder for TFDS.

Resulting layout:

distributed_vae/
 ├── data/                     ← TFDS data_dir
 ├── launch_multiworker.sh
 ├── README.md
 ├── requirements.txt
 └── src/
      ├── __init__.py
      ├── data.py
      ├── model.py
      └── train.py
"""
import os, stat, textwrap
from pathlib import Path

# 1) Define project paths
ROOT     = Path.cwd() / "distributed_vae"
SRC      = ROOT / "src"
DATA_DIR = ROOT / "data"
SRC.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2) src/__init__.py
(SRC / "__init__.py").write_text("# distributed_vae.src package\n")

# 3) src/data.py (loads TFDS into local data_dir)
data_py = textwrap.dedent('''\
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
''')
(SRC / "data.py").write_text(data_py)

# 4) src/model.py
model_py = textwrap.dedent('''\
    import tensorflow as tf

    class Encoder(tf.keras.layers.Layer):
        def __init__(self, latent_dim):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, 2, activation="relu"),
                tf.keras.layers.Conv2D(64, 3, 2, activation="relu"),
                tf.keras.layers.Flatten(),
            ])
            self.mu     = tf.keras.layers.Dense(latent_dim)
            self.logvar = tf.keras.layers.Dense(latent_dim)

        def call(self, x):
            x = self.net(x)
            return self.mu(x), self.logvar(x)

    class Decoder(tf.keras.layers.Layer):
        def __init__(self, latent_dim):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(7*7*64, activation="relu"),
                tf.keras.layers.Reshape((7, 7, 64)),
                tf.keras.layers.Conv2DTranspose(64, 3, 2, padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(32, 3, 2, padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid"),
            ])

        def call(self, z):
            return self.net(z)

    class VAE(tf.keras.Model):
        def __init__(self, latent_dim):
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)
            self.mse     = tf.keras.losses.MeanSquaredError(reduction='none')

        def train_step(self, data):
            x, _ = data
            with tf.GradientTape() as tape:
                mu, logvar = self.encoder(x)
                eps        = tf.random.normal(tf.shape(mu))
                z          = mu + tf.exp(0.5*logvar) * eps
                x_hat      = self.decoder(z)

                recon = tf.reduce_sum(self.mse(x, x_hat), axis=[1,2,3])
                kl    = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
                loss  = tf.reduce_mean(recon + kl)

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {"loss": loss}

        def sample(self, num_samples=100, latent_dim=48):
            z = tf.random.normal((num_samples, latent_dim))
            return self.decoder(z)
''')
(SRC / "model.py").write_text(model_py)

# 5) src/train.py
train_py = textwrap.dedent('''\
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
''')
(SRC / "train.py").write_text(train_py)

# 6) launch_multiworker.sh
launch = textwrap.dedent('''\
    #!/usr/bin/env bash
    # Launch two local workers for the modular VAE example
    HOST=127.0.0.1
    PORT0=12345
    PORT1=12346
    WORKERS=("$HOST:$PORT0" "$HOST:$PORT1")

    for IDX in 0 1; do
      export TF_CONFIG=$(cat <<EOF
    {
      "cluster": { "worker": ["${WORKERS[0]}", "${WORKERS[1]}"] },
      "task":    { "type":"worker", "index": $IDX }
    }
    EOF
      )
      echo "▶ Worker $IDX on port ${WORKERS[$IDX]##*:}"
      CUDA_VISIBLE_DEVICES=$IDX python -u src/train.py &
    done

    wait
''').strip()
( ROOT / "launch_multiworker.sh" ).write_text(launch + "\n")
mode = os.stat(ROOT / "launch_multiworker.sh").st_mode
os.chmod(ROOT / "launch_multiworker.sh", mode | stat.S_IEXEC)

# 7) requirements.txt & README.md
(ROOT / "requirements.txt").write_text("tensorflow==2.16.1\ntensorflow-datasets==4.9.4\n")
readme = textwrap.dedent('''\
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
''').strip()
(ROOT / "README.md").write_text(readme + "\n")

print(f"✅ Modular + data/ scaffold created at {ROOT}")
try:
    import tensorflow_datasets as tfds
    print("➡️  Pre‑downloading MNIST into data/…")
    tfds.load("mnist", split="train", data_dir=str(DATA_DIR), download=True)
    tfds.load("mnist", split="test",  data_dir=str(DATA_DIR), download=True)
    print("✅  MNIST download complete")
except Exception as e:
    print("⚠️  Could not auto‑download MNIST:", e)
