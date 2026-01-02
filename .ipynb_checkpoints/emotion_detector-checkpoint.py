import streamlit as st
from PIL import Image, ImageOps
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
model = load_model('emotion4.keras')
@st.cache_data
def classify(image, model, class_names): 
    # convery image to (224,224)
    image = ImageOps.fit(image, (224,224), Image.Resampling.LANCZOS)
    # convert image to numpy array
    image_array = np.asarray(image)
    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # set model input 
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction) 
    class_name = class_names[index] 
    confidence_score = prediction[0][index]
    
                    
    return class_name, confidence_score
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
class_names = ['angry',
 'Disgust',
 'Fear',
 'Happy',
 'Neutral',
 'Sad',
 'Surprise']

# st.markdown(custom_css, unsafe_allow_html=True)
with tab1: 
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
    if file is not None:
        image = Image.open(file).convert("RGB")
        
        st.image(image)
        class_name, conf_score = classify(image, model, class_names)
        
        # st.write("## {}".format(class_name))
        # st.write("## Score: {}".format(conf_score))
        st.info(f"Emotion Detected: **{class_name}** | Confidence Score: {conf_score:.2f}%")
        

with tab2:
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    
    if picture is not None:
        pic = Image.open(picture).convert("RGB")
        
        st.image(pic)
        submit_btn = st.button('Detect Emotion', type='primary')
        if submit_btn:
            if pic is not None: 
                class_name, conf_score = classify(pic, model, class_names)
            
                st.info(f"Emotion Detected: **{class_name}** | Confidence Score: {conf_score:.2f}%") 
            else: 
                st.error("No image to detect.")