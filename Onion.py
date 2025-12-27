import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# --- SAFETY CHECK ---
try:
    import cv2
except ImportError:
    st.error("CRITICAL: 'opencv-python-headless' is missing from requirements.txt")
    st.stop()

st.set_page_config(page_title="Onion Quality AI", layout="wide")
st.title("🧅 Onion Size Predictor")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Calibration")
    st.info("Using 65mm Cardboard Reference")
    # Set default to 65mm as requested
    ref_size = st.number_input("Reference Circle Dia (mm)", value=65.0)
    st.warning("⚠️ Place the Reference Circle on the LEFT side.")

# --- PROCESSING ENGINE ---
def analyze_circles(uploaded_file, ref_diameter_mm):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        return None, "Error decoding image."

    # 2. Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7) # Slightly stronger blur to smooth paper texture
    
    # 3. Detect All Circles
    # minRadius=15 ensures we catch small onions
    # maxRadius=150 ensures we catch the big 65mm reference
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40,
                               param1=50, param2=30, minRadius=15, maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # 4. Sort: Find the Left-Most Circle (The Reference)
        sorted_circles = circles[0][circles[0][:, 0].argsort()]
        
        # 5. Calculate Scale based on the first circle (The 65mm Ref)
        ref_pixel_width = sorted_circles[0][2] * 2
        pixels_per_mm = ref_pixel_width / ref_diameter_mm
        
        onion_sizes = []
        
        # 6. Draw Visuals
        for i, c in enumerate(sorted_circles):
            # Center coordinates and radius
            center = (c[0], c[1])
            radius = c[2]
            
            if i == 0:
                # This is the Reference (Draw Blue)
                cv2.circle(img, center, radius, (255, 0, 0), 4)
                cv2.putText(img, "REF 65mm", (c[0]-40, c[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # These are Onions (Draw Green)
                diameter_mm = (radius * 2) / pixels_per_mm
                onion_sizes.append(diameter_mm)
                
                cv2.circle(img, center, radius, (0, 255, 0), 3)
                # Label size on the image
                cv2.putText(img, f"{int(diameter_mm)}", (c[0]-15, c[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
        return onion_sizes, img
    else:
        return None, "No circles found. Check lighting and contrast."

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Photo (Reference on Left)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    sizes, result_img = analyze_circles(uploaded_file, ref_size)
    
    if sizes is None:
        st.error(result_img)
    else:
        st.image(result_img, channels="BGR", caption="Blue = Reference | Green = Onions", use_container_width=True)
        
        if len(sizes) > 0:
            df = pd.DataFrame(sizes, columns=['mm'])
            
            # Key Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Sample Count", len(df))
            c2.metric("Avg Diameter", f"{df['mm'].mean():.1f} mm")
            c3.metric("Uniformity (SD)", f"{df['mm'].std():.1f} mm")
            
            # Interactive Chart
            fig = px.histogram(df, x="mm", nbins=12, title="Size Distribution")
            fig.add_vline(x=45, line_dash="dash", line_color="red", annotation_text="Small Limit")
            st.plotly_chart(fig, use_container_width=True)
            
            # Procurement Decision Logic
            small_onions = len(df[df['mm'] < 45])
            medium_onions = len(df[(df['mm'] >= 45) & (df['mm'] <= 60)])
            jumbo_onions = len(df[df['mm'] > 60])
            total = len(df)
            
            st.write("---")
            st.subheader("📊 Lot Grading")
            col_a, col_b, col_c = st.columns(3)
            col_a.info(f"Small (<45mm): {small_onions/total*100:.1f}%")
            col_b.success(f"Medium (45-60mm): {medium_onions/total*100:.1f}%")
            col_c.warning(f"Jumbo (>60mm): {jumbo_onions/total*100:.1f}%")
