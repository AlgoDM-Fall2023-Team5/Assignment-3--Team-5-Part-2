import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import io

# Load the model
model = tf.keras.models.load_model("visual_search_similarity.keras")
label_set = ['Hat', 'Shoes', 'T-Shirt', 'Longsleeve', 'Dress']

def decode_label_prob(y, classes):
    labels = [f'{c}: {p:.2%}' for c, p in zip(classes, y)]
    return labels


st.title("Visual Similarity Finder")

uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])


if uploaded_image is not None:
    try:
        # Read the image using PIL
        image = Image.open(uploaded_image)
        st.write(":green[Image Uploaded Successfully]")
        # st.image(image)
        st.sidebar.write(":blue[Uploaded Image:]")
        st.sidebar.image(image, width=300)
        
        image = image.resize((224, 224))
        # Convert the image to a NumPy array
        image = np.array(image)
        # Preprocess the image
        x = image.astype('float32')
        x /= 255.0  # Normalize pixel values to the range [0, 1]
        # Make a prediction
        x = np.expand_dims(x, axis=0)  # Add a batch dimension
        class_probs = model.predict(x)[0]

        # Decode the prediction results
        labels = decode_label_prob(class_probs, label_set)
        st.header("The Class Probabilities for Image are:")
        st.write(labels, sep="\n")




        

    except Exception as e:
        st.error(f'Failed to load or process the image: {str(e)}')
