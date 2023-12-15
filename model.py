# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Brain tumors MRI",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class

            return key

with st.sidebar:
    st.title("Brain Tumor Detection")
    st.subheader("Brain tumor detection based on brain MRI images")

#st.write("""
#         # Mango Disease Detection with Remedy Suggestion
#         """
#         )
        
model = tf.keras.models.load_model("CNN.h5", compile = False)
#model.compile(learning_rate = 0.001, loss = 'categorical_crossentropy', metrics = 'accuracy')

file = st.file_uploader("", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(file)
    img1 = img.resize((224, 224))
    iArray = tf.keras.preprocessing.image.img_to_array(img1)
    iArray = tf.expand_dims(iArray, 0)
    p = model.predict(iArray)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    st.text(class_names[np.argmax(p)])