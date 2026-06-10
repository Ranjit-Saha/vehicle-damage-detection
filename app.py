import streamlit as st
from PIL import Image
from model_helper import predict

st.title('Vehicle Damage Detection')

# File uploader
uploaded_file = st.file_uploader('Upload the file', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 1. Convert the uploaded memory buffer directly into a PIL Image
    image = Image.open(uploaded_file)

    # 2. Display the image cleanly using the modern parameter layout
    st.image(image, caption='Uploaded file', use_container_width=True)

    # 🔧 FIX: Reset the file pointer so model_helper.py can read the bytes from the start!
    uploaded_file.seek(0)

    # 3. Pass the file object directly to the prediction pipeline
    with st.spinner("Analyzing image..."):
        try:
            prediction = predict(uploaded_file)
            st.info(f"Predicted Class: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
