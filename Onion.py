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

st.set_page_config(page_title="Onion AI Ultimate", layout="wide")
st.title("🧅 Onion AI: Sprout-Proof Mode")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration (Blue Object)")
    st.info("Use a BLUE circle (e.g., bottle cap) as reference.")
    ref_real_size = st.number_input("Reference Size (mm)", value=30.0)

    st.divider()
    st.header("2. Onion Color Tuning")
    # Full HSV Controls
    h_min = st.slider("Hue Min", 0, 179, 0)
    h_max = st.slider("Hue Max", 0, 179, 25)
    s_min = st.slider("Saturation Min", 0, 255, 40)
    v_min = st.slider("Brightness (Value) Min", 0, 255, 40)

    st.divider()
    st.header("3. Shape & Sprout Filters")
    min_area = st.number_input("Min Area (Size)", value=2000)
    
    # NEW: Sprout Removal Slider
    sprout_k_size = st.slider("Sprout Removal Force", 1, 25, 7, step=2,
                              help="Higher value removes thicker sprouts. Must be an odd number.")

    circ_thresh = st.slider("Min Circularity", 0.0, 1.0, 0.60, step=0.05, 
                            help="Lower this if onions with sprouts removed are slightly oval.")
    
    st.divider()
    show_masks = st.checkbox("Show Debug Masks", value=False)

# --- PROCESSING ENGINE ---
def analyze_sprout_proof(uploaded_file, real_ref_mm, h_min, h_max, s_min, v_min, circ_thresh, sprout_k):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: FIND REFERENCE (BLUE) ---
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel_ref = np.ones((5,5), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel_ref)
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_ref:
        return None, "Reference Not Found! Please place a BLUE object in the photo."
    
    ref_contour = max(cnts_ref, key=cv2.contourArea)
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- STEP B: FIND ONIONS & REMOVE SPROUTS ---
    if h_min > h_max: # Red Wrap-Around Logic
        mask1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask_onion = cv2.add(mask1, mask2)
    else:
        mask_onion = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))
    
    # --- NEW: SPROUT REMOVAL OPERATION ---
    # Create an elliptical "eraser" kernel
    sprout_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sprout_k, sprout_k))
    # Apply "Opening" to erase thin protrusions
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, sprout_kernel)
    
    # Standard Cleanup (Fill holes, remove noise)
    kernel_gen = np.ones((5,5), np.uint8)
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel_gen, iterations=2)
    
    mask_onion = cv2.subtract(mask_onion, mask_ref)
    cnts_onion, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_onions = []
    skipped_onions = []
    
    for c in cnts_onion:
        area = cv2.contourArea(c)
        if area > min_area:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= circ_thresh:
                valid_onions.append(c)
            else:
                skipped_onions.append(c)
                
    # --- DRAW RESULTS ---
    result_img = img.copy()
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (255, 0, 0), 4)
    cv2.putText(result_img, "REF", (int(rx)-20, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    sizes = []
    for c in valid_onions:
        ((ox, oy), orad) = cv2.minEnclosingCircle(c)
        dia_mm = (orad * 2) / px_per_mm
        sizes.append(dia_mm)
        cv2.circle(result_img, (int(ox), int(oy)), int(orad), (0, 255, 0), 3)
        cv2.putText(result_img, f"{int(dia_mm)}", (int(ox)-15, int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if show_masks:
        cv2.drawContours(result_img, skipped_onions, -1, (0, 0, 255), 2)

    return sizes, result_img, mask_onion, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Photo (Blue Ref + Onions)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    # Pass new slider value to the function
    result = analyze_sprout_proof(uploaded_file, ref_real_size, h_min, h_max, s_min, v_min, circ_thresh, sprout_k_size)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        st.image(final_img, channels="BGR", caption="Blue=Ref | Green=Valid Onion", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Blue Mask (Ref)", use_container_width=True)
            # This mask shows the result AFTER sprouts are removed
            c2.image(mask_o, caption="Onion Mask (Sprouts Removed)", use_container_width=True)
            
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
