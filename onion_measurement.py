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

st.set_page_config(page_title="Onion AI Green Ref", layout="wide")
st.title("🧅 Onion AI: Production Mode (Green Ref)")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration (Green Cap)")
    st.info("⚠️ Use a SOLID Green Object.")
    ref_real_size = st.number_input("Reference Diameter (mm)", value=30.0)
    
    with st.expander("🔧 Tune Reference Detection"):
        st.write("Adjust these if Green Circle is not detected:")
        ref_h_min = st.slider("Ref Hue Min", 0, 179, 35)
        ref_h_max = st.slider("Ref Hue Max", 0, 179, 90)
        ref_s_min = st.slider("Ref Saturation Min", 0, 255, 80)
        ref_v_min = st.slider("Ref Brightness Min", 0, 255, 70)

    st.divider()
    st.header("2. Onion Detection")
    h_min = st.slider("Onion Hue Min", 0, 179, 160)
    h_max = st.slider("Onion Hue Max", 0, 179, 20)
    s_min = st.slider("Onion Saturation Min", 0, 255, 60)
    v_min = st.slider("Onion Brightness Min", 0, 255, 50)
    
    st.divider()
    st.header("3. Cleaning Tools")
    sprout_k = st.slider("Sprout Eraser Size", 1, 25, 11, step=2)
    min_area = st.number_input("Min Area (Ignore Dirt)", value=4000, step=500)
    
    measure_logic = st.radio("Grading Logic", 
                             ["Min Axis (Width) - Best for Sprouts", 
                              "Max Axis (Length)", 
                              "Enclosing Circle (Standard)"])
    
    show_masks = st.checkbox("Show Debug Masks", value=True)

# --- PROCESSING ENGINE ---
def analyze_production(file_bytes, real_ref_mm, 
                       ref_h_min, ref_h_max, ref_s_min, ref_v_min,
                       h_min, h_max, s_min, v_min, 
                       logic, sprout_k_size, min_area_thresh):
    
    # Decode image from bytes directly
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image. File might be empty."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: REFERENCE (GREEN) ---
    lower_green = np.array([ref_h_min, ref_s_min, ref_v_min])
    upper_green = np.array([ref_h_max, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_ref = np.ones((5,5), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel_ref, iterations=2)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_ERODE, kernel_ref, iterations=1)
    
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_ref: return None, "Green Reference Not Found. Adjust 'Ref Tuning' sliders."
    
    ref_contour = max(cnts_ref, key=cv2.contourArea)
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- STEP B: ONIONS (RED/BROWN) ---
    if h_min > h_max: 
        mask1 = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, 255, 255]))
        mask_onion = cv2.add(mask1, mask2)
    else:
        mask_onion = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, 255, 255]))

    # Fill Holes
    contours_temp, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_onion)
    big_contours = [c for c in contours_temp if cv2.contourArea(c) > (min_area_thresh / 4)]
    cv2.drawContours(mask_filled, big_contours, -1, 255, thickness=cv2.FILLED)
    
    mask_ref_dilated = cv2.dilate(mask_ref, np.ones((15,15), np.uint8), iterations=1)
    mask_final = cv2.subtract(mask_filled, mask_ref_dilated)
    
    # Sprout Removal
    if sprout_k_size > 1:
        kernel_sprout = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sprout_k_size, sprout_k_size))
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel_sprout)
    
    kernel_close = np.ones((5,5), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    cnts_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    result_img = img.copy()
    
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (0, 255, 0), 3)
    cv2.putText(result_img, "REF", (int(rx)-20, int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for c in cnts_final:
        if cv2.contourArea(c) > min_area_thresh:
            if len(c) < 5: continue
            (center, (MA, ma), angle) = cv2.fitEllipse(c)
            axes = sorted([MA, ma])
            minor_axis = axes[0]
            major_axis = axes[1]
            
            if logic == "Min Axis (Width) - Best for Sprouts":
                dia_mm = minor_axis / px_per_mm
                cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 0, 255), 2)
            elif logic == "Max Axis (Length)":
                dia_mm = major_axis / px_per_mm
                cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 255, 255), 2)
            else: 
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                dia_mm = (radius * 2) / px_per_mm
                cv2.circle(result_img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

            sizes.append(dia_mm)
            cv2.putText(result_img, f"{int(dia_mm)}mm", (int(center[0])-20, int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return sizes, result_img, mask_final, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 1. Read bytes ONCE here
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # 2. Pass bytes to function (No fallback logic needed)
    result = analyze_production(file_bytes, ref_real_size, 
                                ref_h_min, ref_h_max, ref_s_min, ref_v_min,
                                h_min, h_max, s_min, v_min, 
                                measure_logic, sprout_k, min_area)

    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        st.image(final_img, channels="BGR", caption="Analyzed Image", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Green Reference Mask", use_container_width=True)
            c2.image(mask_o, caption="Onion Mask", use_container_width=True)
            
        if sizes:
            df = pd.DataFrame(sizes, columns=['mm'])
            m1, m2 = st.columns(2)
            m1.metric("Avg Size", f"{df['mm'].mean():.1f} mm")
            m2.metric("Uniformity", f"{df['mm'].std():.1f} mm")
            fig = px.histogram(df, x="mm", nbins=15, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    elif result:
        st.error(result[1])
