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

st.set_page_config(page_title="Onion AI Precision", layout="wide")
st.title("🧅 Onion AI: High-Precision Mode")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration (Blue Object)")
    st.info("Using 'Convex Hull' to fix hand-drawn circles.")
    ref_real_size = st.number_input("Reference Diameter (mm)", value=60.0)

    st.divider()
    st.header("2. Color & Shadow Tuning")
    # Tuned defaults for your specific images
    h_min = st.slider("Hue Min", 0, 179, 160)
    h_max = st.slider("Hue Max", 0, 179, 20)
    s_min = st.slider("Saturation Min", 0, 255, 65, help="Increase to remove white paper.")
    v_min = st.slider("Brightness Min", 0, 255, 60, help="Increase to remove dark shadows.")

    st.divider()
    st.header("3. Shape Filters")
    min_area = st.number_input("Min Area", value=3000)
    sprout_k = st.slider("Sprout Eraser Size", 1, 21, 11, step=2)
    
    st.markdown("---")
    measure_mode = st.radio("Measurement Mode", ["Max Diameter (Standard)", "Min Diameter (Ignore Tails)"], index=1)
    st.info("💡 'Min Diameter' is best for onions with sprouts.")
    
    show_masks = st.checkbox("Show Debug Masks", value=True)

# --- PROCESSING ENGINE ---
def analyze_precision(uploaded_file, real_ref_mm, h_min, h_max, s_min, v_min, measure_mode, sprout_k):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: ROBUST REFERENCE DETECTION ---
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Close gaps in marker strokes
    kernel_ref = np.ones((7,7), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel_ref, iterations=3)
    
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_ref:
        return None, "Blue Reference Not Found."
    
    # Use Convex Hull to ignore jagged edges of marker
    ref_contour_raw = max(cnts_ref, key=cv2.contourArea)
    ref_contour = cv2.convexHull(ref_contour_raw)
    
    # Measure Reference
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- STEP B: ONION DETECTION ---
    if h_min > h_max: # Red Wrap-Around
        mask1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask_onion = cv2.add(mask1, mask2)
    else:
        mask_onion = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))
    
    # Remove Reference from Onion Mask
    mask_ref_dilated = cv2.dilate(mask_ref, kernel_ref, iterations=5)
    mask_onion = cv2.subtract(mask_onion, mask_ref_dilated)

    # Sprout & Shadow Removal
    sprout_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sprout_k, sprout_k))
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, sprout_kernel)
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel_ref, iterations=2) # Fill holes
    
    cnts_onion, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    result_img = img.copy()
    
    # Draw Reference
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (255, 0, 0), 3)
    cv2.putText(result_img, "REF", (int(rx), int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for c in cnts_onion:
        if cv2.contourArea(c) > min_area:
            # MEASUREMENT LOGIC
            if measure_mode == "Min Diameter (Ignore Tails)":
                if len(c) < 5: continue
                # Fit Ellipse returns (center), (MA, ma), angle
                (center, (MA, ma), angle) = cv2.fitEllipse(c)
                # We take the smaller axis (MA) as the diameter
                # usually width is the grading size for onions
                dia_px = min(MA, ma) 
                # Draw Ellipse
                cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 255, 0), 2)
            else:
                # Standard Circle Fit
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                dia_px = radius * 2
                cv2.circle(result_img, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                center = (cx, cy)

            dia_mm = dia_px / px_per_mm
            sizes.append(dia_mm)
            
            # Label
            cv2.putText(result_img, f"{int(dia_mm)}", (int(center[0])-20, int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return sizes, result_img, mask_onion, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    result = analyze_precision(uploaded_file, ref_real_size, h_min, h_max, s_min, v_min, measure_mode, sprout_k)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        
        st.image(final_img, channels="BGR", caption="Analyzed Image", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Ref Mask (Hull Applied)", use_container_width=True)
            c2.image(mask_o, caption="Onion Mask (Shadows Removed)", use_container_width=True)
            
        if sizes:
            df = pd.DataFrame(sizes, columns=['mm'])
            m1, m2 = st.columns(2)
            m1.metric("Avg Size", f"{df['mm'].mean():.1f} mm")
            m2.metric("Uniformity", f"{df['mm'].std():.1f} mm")
            fig = px.histogram(df, x="mm", nbins=15, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    elif result:
        st.error(result[1])
