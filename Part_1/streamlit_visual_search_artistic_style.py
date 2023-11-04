import streamlit as st
import pickle
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
import scipy as sc
import cv2
import ntpath
import tensorflow as tf
import glob
from PIL import Image

st.header("Visual Search Aristic Style")
# D:\Projects\ADM Assg 3\Assignment-3--Team-5\Part_1\streamlit_visual_search_artistic_style.py
#model = tf.keras.models.load_model("Part_1/visual_search_artistic.keras")

with open('Part_1/image_style_embeddings.pickle', 'rb') as f:
    image_style_embeddings = pickle.load(f)

def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :] # shape -> (batch_size, h, w, d)
    return img

# Visualize the 2D-projection of the embedding space with example images (thumbnails)
def embedding_plot(X, images, thumbnail_sparsity = 0.005, thumbnail_size = 0.3):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    fig, ax = plt.subplots(1, figsize=(12, 12))

    shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < thumbnail_sparsity: continue
        shown_images = np.r_[shown_images, [X[i]]]
        thumbnail = offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=thumbnail_size)
        ax.add_artist(offsetbox.AnnotationBbox(thumbnail, X[i], bboxprops = dict(edgecolor='white'), pad=0.0))

    plt.grid(True)


# embedding_plot(X_tsne, images=list(images.values()))

# # Save the figure as an image
# plt.savefig('embedding_plot.png')

# # Pass the saved image to the st.image() function
# st.image('embedding_plot.png')

# Search in the Embedding Space
def search_by_style(image_style_embeddings, images, reference_image, max_results=10):
    v0 = image_style_embeddings[reference_image]
    distances = {}
    for k,v in image_style_embeddings.items():
        d = sc.spatial.distance.cosine(v0, v)
        distances[k] = d

    sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)
    
    f, ax = plt.subplots(1, max_results, figsize=(16, 8))
    for i, img in enumerate(sorted_neighbors[:max_results]):
        ax[i].imshow(images[img[0]])
        ax[i].set_axis_off()

    plt.show()
    

image_paths = glob.glob('search/images-by-style/*.jpg')
# print(f'Found [{len(image_paths)}] images')
images = {}
for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b,g,r = cv2.split(image) # get b, g, r
    image = cv2.merge([r,g,b]) # switch it to r, g, b
    image = cv2.resize(image, (200, 200))
    images[ntpath.basename(image_path)] = image 

tsne = manifold.TSNE(n_components=2, init='pca', perplexity=10, random_state=0)
X_tsne = tsne.fit_transform( np.array(list(image_style_embeddings.values())) )

# images mostly match the reference style, although not perfectly
#image = Image.open('Part_1/Arn-Van-Gogh-Secondary-1.jpg')

# search_by_style(image_style_embeddings, images, 's_impressionist-02.jpg')
# plt.savefig('style_plot.png')
# st.image('style_plot.png')
# images mostly match the reference style, although not perfectly
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
# st.write(uploaded_image)

if uploaded_image is not None:
    try:
        st.write(":green[Image Uploaded Successfully]")
        st.sidebar.write(":blue[Uploaded Image:]")
        st.sidebar.image(uploaded_image, width=300)
        name = uploaded_image.name
        search_by_style(image_style_embeddings, images, uploaded_image)
        #plt.savefig('search_by_style.png')
        st.header("Matching Images:")
        st.image('search_by_style.png')

        embedding_plot(X_tsne, images=list(images.values()))

        # Save the figure as an image
        
        #plt.savefig('embedding_plot.png')

        # Pass the saved image to the st.image() function
        st.header("2D Embedding Plot:")
        st.image('embedding_plot.png')

    except Exception as e:
      st.error(f'Failed to load or process the image: {str(e)}')