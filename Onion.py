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

st.set_page_config(page_title="Onion AI Pro", layout="wide")
st.title("🧅 Onion AI: Pro Calibration Mode")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration (The Blue Object)")
    st.info("Use a BLUE circle (e.g., plastic bottle cap) as reference.")
    ref_real_size = st.number_input("Reference Size (mm)", value=30.0)
    
    st.divider()
    st.header("2. Onion Filters")
    min_area = st.number_input("Min Area (Size)", value=2000)
    
    # NEW: Circularity Threshold
    # 1.0 is perfect circle. 0.7 allows slightly oval shapes.
    circularity_thresh = st.slider("Min Circularity", 0.0, 1.0, 0.7, step=0.05, 
                                   help="1.0 = Perfect Circle. Lower this if onions are oval.")

    st.divider()
    st.header("3. Color Tuning")
    show_masks = st.checkbox("Show Debug Masks", value=False)
    # Saturation Threshold for Onions
    onion_sat_min = st.slider("Onion Saturation Min", 0, 255, 40, help="Increase if table is being detected as onion.")

# --- ADVANCED PROCESSING ---
def analyze_dual_color(uploaded_file, real_ref_mm, circ_thresh, sat_min):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: FIND REFERENCE (BLUE) ---
    # Blue is typically Hue 100-130 in OpenCV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Clean up Ref Mask
    kernel = np.ones((5,5), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel)
    
    # Find Ref Contours
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_ref:
        return None, "Reference Not Found! Please place a BLUE object (like a bottle cap) in the photo."
    
    # Get the biggest blue object (in case there's noise)
    ref_contour = max(cnts_ref, key=cv2.contourArea)
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- STEP B: FIND ONIONS (RED/BROWN) ---
    # Onions are Low Hue (0-20) and High Hue (160-180) - basically Red/Orange
    # We create two masks and combine them to catch all red shades
    lower_red1 = np.array([0, sat_min, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, sat_min, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_onion = cv2.add(mask1, mask2)
    
    # Clean up Onion Mask
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts_onion, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_onions = []
    skipped_onions = [] # For debugging
    
    for c in cnts_onion:
        area = cv2.contourArea(c)
        if area > min_area:
            # CIRCULARITY CHECK
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            
            # Formula: (4 * pi * Area) / (Perimeter^2)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            if circularity >= circ_thresh:
                valid_onions.append(c)
            else:
                skipped_onions.append(c)
                
    # --- DRAW RESULTS ---
    result_img = img.copy()
    
    # Draw Reference (Blue)
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (255, 0, 0), 4)
    cv2.putText(result_img, "REF", (int(rx)-20, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    sizes = []
    
    # Draw Onions (Green)
    for c in valid_onions:
        ((ox, oy), orad) = cv2.minEnclosingCircle(c)
        dia_mm = (orad * 2) / px_per_mm
        sizes.append(dia_mm)
        
        cv2.circle(result_img, (int(ox), int(oy)), int(orad), (0, 255, 0), 3)
        cv2.putText(result_img, f"{int(dia_mm)}", (int(ox)-15, int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw Skipped (Red Outline - Failed Circularity)
    if show_masks:
        cv2.drawContours(result_img, skipped_onions, -1, (0, 0, 255), 2)

    return sizes, result_img, mask_onion, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Photo (Blue Ref + Onions)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    result = analyze_dual_color(uploaded_file, ref_real_size, circularity_thresh, onion_sat_min)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        
        st.image(final_img, channels="BGR", caption="Blue=Ref | Green=Valid Onion | Red=Not Circular", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Blue Mask (Reference)", use_container_width=True)
            c2.image(mask_o, caption="Red/Brown Mask (Onions)", use_container_width=True)
            
        if sizes:
            df = pd.DataFrame(sizes, columns=['mm'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Count", len(df))
            m2.metric("Avg Size", f"{df['mm'].mean():.1f} mm")
            m3.metric("Uniformity", f"{df['mm'].std():.1f} mm")
            
            fig = px.histogram(df, x="mm", nbins=15, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    elif result:
        st.error(result[1])
