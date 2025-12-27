import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# 1. Safe Import of OpenCV
try:
    import cv2
except ImportError:
    st.error("CRITICAL ERROR: 'opencv-python-headless' is not installed. Please add it to your requirements.txt file.")
    st.stop()

st.set_page_config(page_title="Onion Procurement AI", layout="wide")
st.title("🧅 Onion Size Distribution Predictor")

# --- DEBUGGING STATUS ---
st.sidebar.success("System Status: Online")

def process_image(uploaded_file, ref_diameter_mm):
    try:
        # Convert file to opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            return None, "Error decoding image. Please use a standard JPG or PNG."

        # Grayscale & Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        # Detect Circles
        # Tuned parameters for onions
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=100, param2=30, minRadius=20, maxRadius=150)

        if circles is None:
            return None, "No circles detected. Try a photo with better lighting and clear separation between onions."

        circles = np.uint16(np.around(circles))
        
        # Sort circles to find the reference object (Assume Left-Most is reference)
        sorted_circles = circles[0][circles[0][:, 0].argsort()]
        
        # Get Reference Pixels
        ref_pixel_diameter = sorted_circles[0][2] * 2
        pixels_per_mm = ref_pixel_diameter / ref_diameter_mm
        
        diameters = []
        
        # Draw on image
        # Blue Circle = Reference
        cv2.circle(img, (sorted_circles[0][0], sorted_circles[0][1]), sorted_circles[0][2], (255, 0, 0), 4)
        
        # Green Circles = Onions
        for c in sorted_circles[1:]:
            d_mm = (c[2] * 2) / pixels_per_mm
            diameters.append(d_mm)
            cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 3)
            # Label size
            cv2.putText(img, f"{int(d_mm)}mm", (c[0]-10, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return diameters, img

    except Exception as e:
        return None, f"Processing Error: {str(e)}"

# --- MAIN APP UI ---

ref_size = st.sidebar.number_input("Ref Coin Size (mm)", value=25)

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner('Analyzing onions...'):
        # Reset file pointer to be safe
        uploaded_file.seek(0)
        
        result, processed_img_or_error = process_image(uploaded_file, ref_size)

        if result is None:
            # If result is None, the second variable contains the error message
            st.error(processed_img_or_error)
        else:
            # Success!
            st.image(processed_img_or_error, channels="BGR", caption="Processed Image", use_container_width=True)
            
            # Data Frame
            df = pd.DataFrame(result, columns=['Size_mm'])
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Count", len(df))
            c2.metric("Avg Size", f"{df['Size_mm'].mean():.1f} mm")
            c3.metric("Std Dev", f"{df['Size_mm'].std():.1f} mm")
            
            # Chart
            fig = px.histogram(df, x="Size_mm", nbins=15, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality Check
            st.subheader("Grading")
            small_pct = (len(df[df['Size_mm'] < 45]) / len(df)) * 100
            if small_pct > 10:
                st.warning(f"⚠️ High percentage of small onions: {small_pct:.1f}%")
            else:
                st.success("✅ Lot looks good!")
