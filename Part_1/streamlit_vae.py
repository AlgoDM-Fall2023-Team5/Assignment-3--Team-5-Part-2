import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
ds = tfp.distributions
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter("ignore")
import streamlit as st

@st.cache
def get_all_embeddings(data, vae):
    all_embeddings = []
    for image in data:
        image = image.reshape(1, DIMS)
        embeddings = vae.encode(image).loc[0].numpy()
        all_embeddings.append(embeddings)
    return np.array(all_embeddings)

@st.cache
def query(image_id, k):
    query_embedding = all_embeddings[image_id]
    distances = np.zeros(len(all_embeddings))
    for i, e in enumerate(all_embeddings):
        distances[i] = np.linalg.norm(query_embedding - e)
    return np.argpartition(distances, k)[:k]


st.set_page_config(layout='wide')
st.title("Visual Search using VAE")

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

TRAIN_BUF = 60000
BATCH_SIZE = 512
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)

# split dataset
train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)
# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

fig, ax = plt.subplots(10, 11, figsize=(12, 10), gridspec_kw={'width_ratios': [2] + [1]*10})
img_idx = 0
for i in range(10):
    ax[i, 0].axis('off')
    ax[i, 0].text(0.5, 0.5, fashion_mnist_labels[i])
    
    class_indexes = [k for k, n in enumerate(y_train) if n == i]
    for j in range(10):
        ax[i, j+1].imshow(1 - train_images[class_indexes[j], :])
        ax[i, j+1].axis('off')

# Define the VAE model
class VAE(tf.keras.Model):

    DIMS = (28, 28, 1)
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        self.__dict__.update(kwargs)

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

    def encode(self, x):
        mu, sigma = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def reconstruct(self, x):
        mu, _ = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return self.decode(mu)

    def decode(self, z):
        return self.dec(z)

    def compute_loss(self, x):
        q_z = self.encode(x)
        z = q_z.sample()
        x_recon = self.decode(z)
        p_z = ds.MultivariateNormalDiag(
          loc=[0.] * z.shape[-1], scale_diag=[1.] * z.shape[-1]
          )
        kl_div = ds.kl_divergence(q_z, p_z)
        latent_loss = tf.reduce_mean(tf.maximum(kl_div, 0))
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - x_recon), axis=0))

        return recon_loss, latent_loss

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    def call(self, inputs):  # Define the forward pass here
        return self.decode(self.encode(inputs).sample())
        
    @tf.function
    def train(self, train_x):
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        
        
N_Z = 2
encoder = [
    tf.keras.layers.InputLayer(input_shape=DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=N_Z*2),
]

decoder = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]


# Model training
optimizer = tf.keras.optimizers.legacy.Adam(1e-3)
model = VAE(
    enc = encoder,
    dec = decoder,
    optimizer = optimizer,
)

# Create a grid over the semantic space

nx = ny = 10
meshgrid = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))
meshgrid = np.array(meshgrid).reshape(2, nx*ny).T
x_grid = model.decode(meshgrid)
x_grid = x_grid.numpy().reshape(nx, ny, 28,28, 1)

def get_all_embeddings(data, vae):
    all_embeddings = []
    for image in data:
        image = image.reshape(1, *DIMS)  # Reshape your image as needed
        embeddings = vae.encode(image).loc[0].numpy()  # Assuming 'loc' contains the mean of the embeddings
        all_embeddings.append(embeddings)
    return np.array(all_embeddings)


vae = VAE(enc=encoder, dec=decoder, optimizer = optimizer)
vae.build((None, *DIMS))
vae.load_weights('Part_1/vae_model.h5')

all_embeddings = get_all_embeddings(train_images, vae)

def query(image_id, k):
    query_embedding = all_embeddings[image_id]
    distances = np.zeros(len(all_embeddings))
    for i, e in enumerate(all_embeddings):
        distances[i] = np.linalg.norm(query_embedding - e)
    return np.argpartition(distances, k)[:k]

# User inputs
query_image_id = st.number_input("Enter Image ID (0 - 100)", min_value=0, max_value=100, value=15, step=1)
k = st.number_input("Number of Similar Images (1 - 6)", min_value=1, max_value=10, value=6, step=1)


# Perform the query based on user input
idx = query(query_image_id, k=k)

# Create a plot for the results
st.write('Query Results:')
fig, ax = plt.subplots(1, k, figsize=(k*2, 2))
# Display the selected images
for i in range(k):
    ax[i].imshow(1 - train_images[idx[i], :])
    ax[i].axis('off')

# Save the figure as a PDF file (optional)
st.pyplot(fig)
