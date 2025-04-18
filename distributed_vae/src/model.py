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
