import io

import requests
from PIL import Image
from deeplab_infer import publaynet_dataset_information, vis_segmentation
import tensorflow as tf


import streamlit as st

# interact with FastAPI endpoint
model_dir = "/home/cuongph14/Downloads/panoptic_deeplab/resnet16_pd-20220304T195402Z-001/resnet16_pd"
DATASET_INFO = publaynet_dataset_information()
LOADED_MODEL = tf.saved_model.load(model_dir)


def process(image):

    output = LOADED_MODEL(tf.cast(image, tf.uint8))
    fig = vis_segmentation(image, output['panoptic_pred'][0], DATASET_INFO)

    return fig


# construct UI layout
st.title("Panoptic Deeplab")

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Get segmentation map"):

    col1= st.columns(1)

    if input_image:
        original_image = Image.open(input_image).convert("RGB")
        segments = process(original_image)
        col1[0].header("Original")
        with col1[0]:
            st.pyplot(segments)

    else:
        # handle case with no image
        st.write("Insert an image!")