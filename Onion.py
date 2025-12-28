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

st.set_page_config(page_title="Onion AI Manager", layout="wide")
st.title("🧅 Onion Procurement Assistant (Color Mode)")

# --- SIDEBAR TUNING ---
with st.sidebar:
    st.header("Calibration")
    ref_size = st.number_input("Reference Coin Size (mm)", value=25.0)
    
    st.divider()
    st.header("Color Tuning (HSV)")
    st.info("Adjust these to isolate the Onion color.")
    
    # Defaults tuned for Brown/Red Onions
    # Hue: 0-179 covers all colors. Onions are usually Red/Orange (low hue)
    hue_min = st.slider("Hue Min", 0, 179, 0)
    hue_max = st.slider("Hue Max", 0, 179, 179)
    # Saturation: 0 is gray/white, 255 is vibrant color
    sat_min = st.slider("Saturation Min", 0, 255, 30, help="Increase this if it detects the white table.") 
    # Value: Brightness
    val_min = st.slider("Brightness Min", 0, 255, 0)
    
    min_area = st.number_input("Min Object Area", value=3000, step=500, help="Increase to ignore small dirt specks.")

# --- COLOR PROCESSING ENGINE ---
def analyze_color(uploaded_file, ref_diameter_mm, h_min, h_max, s_min, v_min):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        return None, "Error decoding image."

    # 2. Convert to HSV Color Space
    # HSV (Hue, Saturation, Value) is better for filtering than RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Create Color Mask
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, 255, 255])
    
    # This creates a Black & White image (White = Onion, Black = Table)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 4. Clean up Mask (Remove noise)
    kernel = np.ones((5,5), np.uint8)
    # "Open" removes white dots (noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # "Dilate" fills holes inside the onion
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # 5. Find Contours on the Mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, "No objects detected. Try lowering 'Saturation Min'."

    # 6. Filter Small Objects
    valid_objects = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            valid_objects.append(c)
            
    if not valid_objects:
        return None, "Objects too small. Decrease 'Min Object Area'."

    # Sort Left-to-Right (Assume Leftmost is Reference)
    valid_objects = sorted(valid_objects, key=lambda c: cv2.boundingRect(c)[0])
    
    # 7. Measurement Logic
    onion_sizes = []
    result_img = img.copy()
    
    # Draw Mask preview for debugging
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Assume first object is Reference
    ref_contour = valid_objects[0]
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    ref_px_width = ref_radius * 2
    pixels_per_mm = ref_px_width / ref_diameter_mm
    
    for i, c in enumerate(valid_objects):
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)
        center = (int(cx), int(cy))
        radius = int(radius)
        diameter_mm = (radius * 2) / pixels_per_mm
        
        if i == 0:
            # Reference
            cv2.circle(result_img, center, radius, (255, 0, 0), 3)
            cv2.putText(result_img, "REF", (center[0]-20, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            # Onions
            onion_sizes.append(diameter_mm)
            cv2.circle(result_img, center, radius, (0, 255, 0), 3)
            cv2.putText(result_img, f"{int(diameter_mm)}", (center[0]-15, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    return onion_sizes, result_img, mask_bgr

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    # Run Analysis
    result = analyze_color(uploaded_file, ref_size, hue_min, hue_max, sat_min, val_min)
    
    if result and len(result) == 3: # Success
        sizes, final_img, debug_mask = result
        
        # Show Results
        col_A, col_B = st.columns(2)
        with col_A:
            st.image(final_img, channels="BGR", caption="Final Detection (Green Circles)", use_container_width=True)
        with col_B:
            st.image(debug_mask, channels="BGR", caption="Computer Vision Mask (White=Object)", use_container_width=True)
            st.info("👆 If the white shape has holes, LOWER 'Saturation Min'. If the background is white, INCREASE it.")
        
        if len(sizes) > 0:
            df = pd.DataFrame(sizes, columns=['mm'])
            c1, c2, c3 = st.columns(3)
            c1.metric("Count", len(df))
            c2.metric("Avg Size", f"{df['mm'].mean():.1f} mm")
            c3.metric("Uniformity", f"{df['mm'].std():.1f} mm")
            
            fig = px.histogram(df, x="mm", nbins=10, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    elif result: # Error Message
        st.error(result[1])
        st.warning("Tip: If the image is black, try sliding 'Saturation Min' to 20.")
