import io

import requests
from PIL import Image
from deeplab_infer import process
import streamlit as st




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
