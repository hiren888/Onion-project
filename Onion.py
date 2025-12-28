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

st.set_page_config(page_title="Onion AI Production", layout="wide")
st.title("🧅 Onion AI: Production Mode")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration")
    st.info("⚠️ Use a SOLID Blue Object (Cap/Chip), not a drawing.")
    ref_real_size = st.number_input("Reference Diameter (mm)", value=30.0) # Set this to your cap size

    st.divider()
    st.header("2. Detection Tuning")
    # Updated defaults for your shiny onions
    h_min = st.slider("Hue Min", 0, 179, 160)
    h_max = st.slider("Hue Max", 0, 179, 20)
    # Lower saturation slightly to catch shiny parts, but keep Value high
    s_min = st.slider("Saturation Min", 0, 255, 60)
    v_min = st.slider("Brightness Min", 0, 255, 50)
    
    st.divider()
    st.header("3. Advanced Filters")
    min_area = st.number_input("Min Area", value=3500)
    
    # NEW: Measurement Mode
    measure_logic = st.radio("Grading Logic", 
                             ["Min Axis (Width) - Best for Sprouts", 
                              "Max Axis (Length)", 
                              "Enclosing Circle (Standard)"])
    
    show_masks = st.checkbox("Show Debug Masks", value=True)

# --- PROCESSING ENGINE ---
def analyze_production(uploaded_file, real_ref_mm, h_min, h_max, s_min, v_min, logic):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: REFERENCE (BLUE) ---
    lower_blue = np.array([90, 120, 50])
    upper_blue = np.array([140, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Fill holes in reference (fix marker gaps)
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_ref: return None, "Blue Reference Not Found."
    
    # Get largest blue object
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

    # --- NEW: FLOOD FILL HOLES (The Fix for Shiny Skin) ---
    # Find contours of the 'swiss cheese' mask
    contours_temp, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw them filled in White (255) on a new mask
    mask_filled = np.zeros_like(mask_onion)
    cv2.drawContours(mask_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    
    # Separate Reference from Onions
    mask_ref_dilated = cv2.dilate(mask_ref, np.ones((15,15), np.uint8), iterations=1)
    mask_final = cv2.subtract(mask_filled, mask_ref_dilated)
    
    # Open operation to remove thin sprouts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
    
    cnts_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    result_img = img.copy()
    
    # Draw Reference
    ((rx, ry), rr) = cv2.minEnclosingCircle(ref_contour)
    cv2.circle(result_img, (int(rx), int(ry)), int(rr), (255, 0, 0), 3)
    cv2.putText(result_img, "REF", (int(rx), int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for c in cnts_final:
        if cv2.contourArea(c) > 3500: # Filter small noise
            
            # FIT ELLIPSE (Best for shape analysis)
            if len(c) < 5: continue
            (center, (MA, ma), angle) = cv2.fitEllipse(c)
            
            # MA = Minor Axis (Width), ma = Major Axis (Length)
            # OpenCV fitEllipse returns (MA, ma) but order varies, so we sort them
            axes = sorted([MA, ma])
            minor_axis = axes[0]
            major_axis = axes[1]
            
            if logic == "Min Axis (Width) - Best for Sprouts":
                dia_mm = minor_axis / px_per_mm
                # Draw Ellipse (Green)
                cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 255, 0), 2)
                # Draw Width Line (Red) to show user what we measured
                # (Visualizing the minor axis is complex math, simplifying for UI)
                
            elif logic == "Max Axis (Length)":
                dia_mm = major_axis / px_per_mm
                cv2.ellipse(result_img, (center, (MA, ma), angle), (0, 255, 255), 2)
                
            else: # Enclosing Circle
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                dia_mm = (radius * 2) / px_per_mm
                cv2.circle(result_img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

            sizes.append(dia_mm)
            
            # Label
            cv2.putText(result_img, f"{int(dia_mm)}mm", (int(center[0])-20, int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return sizes, result_img, mask_final, mask_ref

# --- UI ---
uploaded_file = st.file_uploader("Upload Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file.seek(0)
    result = analyze_production(uploaded_file, ref_real_size, h_min, h_max, s_min, v_min, measure_logic)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        
        st.image(final_img, channels="BGR", caption="Analyzed Image", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Reference Mask", use_container_width=True)
            c2.image(mask_o, caption="Onion Mask (Holes Filled)", use_container_width=True)
            
        if sizes:
            df = pd.DataFrame(sizes, columns=['mm'])
            m1, m2 = st.columns(2)
            m1.metric("Avg Size", f"{df['mm'].mean():.1f} mm")
            m2.metric("Uniformity", f"{df['mm'].std():.1f} mm")
            
            fig = px.histogram(df, x="mm", nbins=15, title="Size Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
    elif result:
        st.error(result[1])
