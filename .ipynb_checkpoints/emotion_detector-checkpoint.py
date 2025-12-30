import streamlit as st 
import pandas as pd 
import numpy as np

st.set_page_config(layout='wide', page_title='Basic Layout')
st.title('Layout and State')

# sidebar 
with st.sidebar: 
    st.header('Settings')
    theme = st.selectbox('Choose theme', ['Light', 'Dark', 'System'])

tab1, tab2, tab3 = st.tabs(['Analytics', 'Data', 'ML Prediction']) 

with tab1: 
    st.subheader('Analytics Dashboard')
    col1, col2, col3 = st.columns(3)
    col1.metric('Revenue', '$12000', '+12%')
    col2.metric('User Satisfaction', '$12000', '-5%')
    col3.metric('Rating', '4.5/5', '.3%')


with tab2: 
    st.subheader('Config')
    st.checkbox('Enable Advanced Mode')
    
with tab3: 
    st.subheader('Predict')
    st.checkbox('AI Mode')

# expander
st.divider() 
with st.expander('See Explanation'): 
    st.write('This content is hidden by default.')

st.divider() 
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable, width=500)

if picture:
    st.image(picture)

st.divider()   
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.image(uploaded_file, width=500)
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

# sessions state
st.divider() 
st.subheader('State Management')
st.write('Variables reset store in state session.')

if 'count' not in st.session_state:
    st.session_state.count=0

col_a, col_b = st.columns(2)
with col_a: 
    if st.button('Increment Counter'):
        st.session_state.count+=1
with col_b: 
    st.write(f'Current count: **{st.session_state.count}**')

st.divider()
st.header('Dataframe')
df = pd.DataFrame({
    'Name': ['Aisy', 'Muadz', 'Potato', 'Irfan'],
    'Age': [99, 123, 1000, 5],
    'City': ['Atlantis', 'Mars', 'Tanjung Rambutan', 'Jepang']
})

col1, col2 = st.columns(2) 
with col1: 
    st.write('Interactive Dataframe')
    st.dataframe(df)
    
with col2: 
    st.write('Static Dataframe')
    st.table(df)

# interactive widgets 
st.divider() 
st.header('Interactive Widgets')

name = st.text_input('Enter your good name', 'amanda')
if st.button('Say annyeong'): 
    st.success(f"Hello, {name}!")

age = st.slider('Select your age:', 0,100,25)
st.write(f"How young: {age}")


# simple charts
st.divider()
st.header('Simple chart')
chart_data = pd.DataFrame((np.random.rand(20,3)), columns=['A','B','C'])
st.line_chart(chart_data)