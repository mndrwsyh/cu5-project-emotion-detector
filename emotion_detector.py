import streamlit as st
from PIL import Image
from PIL.ImageFilter import *
from PIL import ImageEnhance
#import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from base64 import b64encode
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
##############################################################################################################
st.set_page_config(page_title="emotion detector", page_icon="😃", layout="centered", initial_sidebar_state="collapsed")
#st.set_option('deprecation.showPyplotGlobalUse', False)
##############################################################################################################
model = load_model('emotion2.keras')
@st.cache_data
def display_histogram(img):
        img = np.array(img)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[0].set_title("Image")
        ax[1].hist(img.ravel(), bins=256, histtype="step", color="black")
        ax[1].set_title("Histogram")
        plt.suptitle("Image and its Histogram")
        st.pyplot(fig)
@st.cache_data
def luminosite(img, pourcentage):
    return ImageEnhance.Brightness(img).enhance(1 + pourcentage)
@st.cache_data
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href
##############################################################################################################
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center;">
<svg width="100%" viewBox="0 0 700 100" xmlns="http://www.w3.org/2000/svg">
    <text 
        x="50%" 
        y="50%" 
        dominant-baseline="middle" 
        text-anchor="middle"
        font-size="80" 
        stroke="#9ABDDC" 
        fill="none" 
        stroke-width="1.5">
        EMOTION DETECTOR
    </text>
</svg>
</div>
""", 
unsafe_allow_html=True)

# custom_css = """
# <style>
# /* Inactive tabs */
# .stTabs [data-baseweb="tab"] {
#     color: #9ABDDC; /* text color */
#     font-family: 'Courier New', Courier, monospace;
#     border-bottom: 2px solid transparent; /* default line */
# }

# /* Active tab */
# .stTabs [aria-selected="true"] {
#     color: #3A5D9C; /* active text color */
#     font-family: 'Courier New', Courier, monospace;
#     border-bottom: 2px solid #FF4B4B; /* active tab underline color */
# }
# </style>
# """
tab1, tab2 = st.tabs(['Upload An Image 🖼️', 'Take A Picture 📸'])

# st.markdown(custom_css, unsafe_allow_html=True)
with tab1: 
    with st.form("UPLOAD IMAGE"):
        image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
        if image:
            imgg = Image.open(image)
            final_img = imgg
            old = imgg
            gray_img = imgg.convert("L")
            
            st.image(imgg)
        submit_btn = st.form_submit_button('Detect Emotion', type='primary')
        if submit_btn:
            img = imgg.convert("RGB")
    
            # Resize to EfficientNet input size
            img = img.resize((224, 224))
            
            # Convert to numpy
            img_array = np.array(img)
            
            # Convert to float & preprocess
            img_array = preprocess_input(img_array)
            
            # Add batch dimension
            input_arr = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(input_arr)
            st.info(f"Emotion Detected: {prediction}")
            st.snow()
            st.info('This person is feeling sad 😢')

with tab2:
    with st.form("TAKE A PICTURE"):   
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)
        
        if picture:
            st.image(picture)
        submit_btn = st.form_submit_button('Detect Emotion', type='primary')
    if submit_btn:
        st.snow()
        st.info('This person is feeling sad 😢')