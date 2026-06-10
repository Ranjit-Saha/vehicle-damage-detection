import streamlit as st
from PIL import Image
from model_helper import predict

# Set a wide page layout to make it look like a modern dashboard dashboard
st.set_page_config(page_title="AI Auto Inspector", layout="wide")

st.markdown("# 🚗 Vehicle Damage Detection System")
st.markdown("### *Automated Deep Learning Inspection Portal*")
st.write("---")

# File uploader
uploaded_file = st.file_uploader('Upload a vehicle image for instant structural assessment',
                                 type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 1. Reset pointer and convert buffer to PIL Image
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)

    # 2. Create side-by-side columns (Left: Image, Right: Analysis Results)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Uploaded Asset")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 AI Structural Analysis")

        # Reset file pointer again so the backend model can read it freshly
        uploaded_file.seek(0)

        with st.spinner("Executing ResNet50 Inference Pipeline..."):
            try:
                # 🔧 SYNC FIX: Unpack both the class string and the mathematical confidence decimal float
                prediction, real_confidence = predict(uploaded_file)

                # Assign dynamic colors and emojis based on the predicted class string
                if "Normal" in prediction:
                    bg_color = "#d4edda"  # Soft Green
                    text_color = "#155724"
                    border_color = "#c3e6cb"
                    status_emoji = "✅ UNDAMAGED"
                elif "Breakage" in prediction:
                    bg_color = "#fff3cd"  # Soft Yellow
                    text_color = "#856404"
                    border_color = "#ffeeba"
                    status_emoji = "⚠️ COMPONENT BROKEN"
                else:  # "Crushed"
                    bg_color = "#f8d7da"  # Soft Red
                    text_color = "#721c24"
                    border_color = "#f5c6cb"
                    status_emoji = "💥 SEVERE CRUSH IMPACT"

                # Render an enterprise-grade HTML Status Card widget
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color}; 
                        color: {text_color}; 
                        border: 1px solid {border_color}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        margin-bottom: 20px;
                        text-align: center;
                    ">
                        <h2 style="margin: 0; font-size: 24px;">{status_emoji}</h2>
                        <p style="margin: 10px 0 0 0; font-size: 18px; font-weight: bold;">
                            Detected Target State: <span style="text-transform: uppercase;">{prediction}</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 🔧 SYNC FIX: Feed the dynamic Softmax probability value directly into the progress bar UI
                st.write(f"**Model Confidence Profile: {real_confidence * 100:.1f}%**")
                st.progress(real_confidence)  # Progress expects a float between 0.0 and 1.0
                st.caption("Live statistical confidence rating generated via PyTorch Softmax distribution layer.")

            except Exception as e:
                st.error(f"System Inference Error: {e}")
