import streamlit as st
from PIL import Image, ImageOps
from PIL.ImageFilter import *
from PIL import ImageEnhance
#import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from base64 import b64encode
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
##############################################################################################################
st.set_page_config(page_title="emotion detector", page_icon="😃", layout="centered", initial_sidebar_state="collapsed")
#st.set_option('deprecation.showPyplotGlobalUse', False)
##############################################################################################################
model = load_model('emotion4.keras')
@st.cache_data
def set_background(color): 
    style = f"""
        <style>
        .stApp {{
            background-color: {color}
        }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)
    
    
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
    index = np.argmax(prediction[0]) 
    class_name = class_names[index] 
    confidence_score = prediction[0][index]
    
                    
    return class_name, confidence_score
##############################################################################################################
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center;">
    <svg width="100%" viewBox="0 0 700 100" xmlns="http://www.w3.org/2000/svg" viewBox="0 8.84000015258789 516.719970703125 37.029998779296875" data-asc="0.9052734375"><defs/><g rg id="svgGroup" stroke-linecap="round" fill-rule="evenodd" font-size="12pt" stroke="#366d8f" stroke-width="0.25mm" fill="none"><g transform="translate(0, 0)"><path d="M0 45.26L0 9.47L25.88 9.47L25.88 13.70L4.74 13.70L4.74 24.66L24.54 24.66L24.54 28.86L4.74 28.86L4.74 41.04L26.71 41.04L26.71 45.26L0 45.26ZM33.11 45.26L33.11 9.47L40.23 9.47L48.71 34.81Q49.88 38.35 50.42 40.11Q51.03 38.16 52.32 34.38L60.89 9.47L67.26 9.47L67.26 45.26L62.70 45.26L62.70 15.31L52.29 45.26L48.02 45.26L37.67 14.79L37.67 45.26L33.11 45.26ZM73.46 27.83Q73.46 18.92 78.25 13.88Q83.03 8.84 90.60 8.84Q95.56 8.84 99.54 11.21Q103.52 13.57 105.60 17.81Q107.69 22.05 107.69 27.42Q107.69 32.86 105.49 37.16Q103.30 41.46 99.27 43.66Q95.24 45.87 90.58 45.87Q85.52 45.87 81.54 43.43Q77.56 40.99 75.51 36.77Q73.46 32.54 73.46 27.83M78.34 27.91Q78.34 34.38 81.82 38.10Q85.30 41.82 90.55 41.82Q95.90 41.82 99.35 38.06Q102.81 34.30 102.81 27.39Q102.81 23.02 101.33 19.76Q99.85 16.50 97.01 14.71Q94.17 12.92 90.63 12.92Q85.60 12.92 81.97 16.37Q78.34 19.82 78.34 27.91ZM122.90 45.26L122.90 13.70L111.11 13.70L111.11 9.47L139.48 9.47L139.48 13.70L127.64 13.70L127.64 45.26L122.90 45.26ZM145.14 45.26L145.14 9.47L149.88 9.47L149.88 45.26L145.14 45.26ZM156.79 27.83Q156.79 18.92 161.57 13.88Q166.36 8.84 173.93 8.84Q178.88 8.84 182.86 11.21Q186.84 13.57 188.93 17.81Q191.02 22.05 191.02 27.42Q191.02 32.86 188.82 37.16Q186.62 41.46 182.59 43.66Q178.56 45.87 173.90 45.87Q168.85 45.87 164.87 43.43Q160.89 40.99 158.84 36.77Q156.79 32.54 156.79 27.83M161.67 27.91Q161.67 34.38 165.15 38.10Q168.63 41.82 173.88 41.82Q179.22 41.82 182.68 38.06Q186.13 34.30 186.13 27.39Q186.13 23.02 184.66 19.76Q183.18 16.50 180.33 14.71Q177.49 12.92 173.95 12.92Q168.92 12.92 165.30 16.37Q161.67 19.82 161.67 27.91ZM197.07 45.26L197.07 9.47L201.93 9.47L220.73 37.57L220.73 9.47L225.27 9.47L225.27 45.26L220.41 45.26L201.61 17.14L201.61 45.26L197.07 45.26ZM247.12 45.26L247.12 9.47L259.45 9.47Q263.62 9.47 265.82 9.99Q268.90 10.69 271.07 12.55Q273.90 14.94 275.31 18.66Q276.71 22.39 276.71 27.17Q276.71 31.25 275.76 34.40Q274.80 37.55 273.32 39.61Q271.83 41.67 270.06 42.86Q268.29 44.04 265.78 44.65Q263.28 45.26 260.03 45.26L247.12 45.26M251.86 41.04L259.50 41.04Q263.04 41.04 265.05 40.38Q267.07 39.72 268.26 38.53Q269.95 36.84 270.89 34.00Q271.83 31.15 271.83 27.10Q271.83 21.48 269.98 18.47Q268.14 15.45 265.50 14.43Q263.60 13.70 259.38 13.70L251.86 13.70L251.86 41.04ZM283.33 45.26L283.33 9.47L309.20 9.47L309.20 13.70L288.06 13.70L288.06 24.66L307.86 24.66L307.86 28.86L288.06 28.86L288.06 41.04L310.03 41.04L310.03 45.26L283.33 45.26ZM325.68 45.26L325.68 13.70L313.89 13.70L313.89 9.47L342.26 9.47L342.26 13.70L330.42 13.70L330.42 45.26L325.68 45.26ZM347.22 45.26L347.22 9.47L373.10 9.47L373.10 13.70L351.95 13.70L351.95 24.66L371.75 24.66L371.75 28.86L351.95 28.86L351.95 41.04L373.93 41.04L373.93 45.26L347.22 45.26ZM406.01 32.71L410.74 33.91Q409.25 39.75 405.38 42.81Q401.51 45.87 395.92 45.87Q390.14 45.87 386.51 43.52Q382.89 41.16 380.99 36.69Q379.10 32.23 379.10 27.10Q379.10 21.51 381.24 17.35Q383.37 13.18 387.32 11.02Q391.26 8.86 396.00 8.86Q401.37 8.86 405.03 11.60Q408.69 14.33 410.13 19.29L405.47 20.39Q404.22 16.48 401.86 14.70Q399.49 12.92 395.90 12.92Q391.77 12.92 389.00 14.89Q386.23 16.87 385.11 20.20Q383.98 23.54 383.98 27.08Q383.98 31.64 385.31 35.05Q386.65 38.45 389.45 40.14Q392.26 41.82 395.53 41.82Q399.51 41.82 402.27 39.53Q405.03 37.23 406.01 32.71ZM425.68 45.26L425.68 13.70L413.89 13.70L413.89 9.47L442.26 9.47L442.26 13.70L430.42 13.70L430.42 45.26L425.68 45.26ZM444.78 27.83Q444.78 18.92 449.56 13.88Q454.35 8.84 461.91 8.84Q466.87 8.84 470.85 11.21Q474.83 13.57 476.92 17.81Q479.00 22.05 479.00 27.42Q479.00 32.86 476.81 37.16Q474.61 41.46 470.58 43.66Q466.55 45.87 461.89 45.87Q456.84 45.87 452.86 43.43Q448.88 40.99 446.83 36.77Q444.78 32.54 444.78 27.83M449.66 27.91Q449.66 34.38 453.14 38.10Q456.62 41.82 461.87 41.82Q467.21 41.82 470.67 38.06Q474.12 34.30 474.12 27.39Q474.12 23.02 472.64 19.76Q471.17 16.50 468.32 14.71Q465.48 12.92 461.94 12.92Q456.91 12.92 453.28 16.37Q449.66 19.82 449.66 27.91ZM485.18 45.26L485.18 9.47L501.05 9.47Q505.83 9.47 508.33 10.44Q510.82 11.40 512.30 13.84Q513.79 16.28 513.79 19.24Q513.79 23.05 511.33 25.66Q508.86 28.27 503.71 28.98Q505.59 29.88 506.57 30.76Q508.64 32.67 510.50 35.52L516.72 45.26L510.77 45.26L506.03 37.82Q503.96 34.59 502.61 32.89Q501.27 31.18 500.21 30.49Q499.15 29.81 498.05 29.54Q497.24 29.37 495.41 29.37L489.92 29.37L489.92 45.26L485.18 45.26M489.92 25.27L500.10 25.27Q503.34 25.27 505.18 24.60Q507.01 23.93 507.96 22.45Q508.91 20.97 508.91 19.24Q508.91 16.70 507.07 15.06Q505.22 13.43 501.25 13.43L489.92 13.43L489.92 25.27Z"/></g></g></svg>

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
class_names = ['Angry',
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
        if class_name == 'Sad':
            set_background('#e0fbff')
        elif class_name == 'Happy': 
            set_background('#fdffe0')
        elif class_name == 'Angry': 
            set_background('#ffe0e0')
        elif class_name == 'Fear': 
            set_background('#fff3e0')
        elif class_name == 'Disgust': 
            set_background('#e0ffe7')
        elif class_name == 'Neutral': 
            set_background('#f2f2f2')
        elif class_name == 'Surprise': 
            set_background('#f9e0ff')
        st.write(f"Emotion Detected: **{class_name}** | Confidence Score: {conf_score:.2f}%")
    else: 
        set_background('white')
        

with tab2:
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)
    
    if picture is not None:
        pic = Image.open(picture).convert("RGB")
        
        st.image(pic)
        # submit_btn = st.button('Detect Emotion', type='primary')
        # if submit_btn:
        # if pic is not None: 
        class_name, conf_score = classify(pic, model, class_names)

        if class_name == 'Sad':
            set_background('#e0fbff')
        elif class_name == 'Happy': 
            set_background('#fdffe0')
        elif class_name == 'Angry': 
            set_background('#ffe0e0')
        elif class_name == 'Fear': 
            set_background('#fff3e0')
        elif class_name == 'Disgust': 
            set_background('#e0ffe7')
        elif class_name == 'Neutral': 
            set_background('#f2f2f2')
        elif class_name == 'Surprise': 
            set_background('#f9e0ff')
        st.write(f"Emotion Detected: **{class_name}** ({conf_score:.2f}%)") 
        # else: 
        #     st.error("No image to detect.")