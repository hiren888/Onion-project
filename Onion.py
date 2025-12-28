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
st.title("🧅 Onion AI: Ultimate Mode")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("1. Calibration (Blue Object)")
    st.info("Use a BLUE circle (e.g., bottle cap) as reference.")
    ref_real_size = st.number_input("Reference Size (mm)", value=30.0)

    st.divider()
    st.header("2. Onion Color Tuning")
    st.info("Adjust these to isolate the onions.")
    
    # Full HSV Controls
    # Onions are often Hue 0-20 or 160-180. 
    # If Min > Max, the app automatically handles the "Red Wrap-around".
    h_min = st.slider("Hue Min", 0, 179, 0, help="0 is Red, 60 Green, 120 Blue.")
    h_max = st.slider("Hue Max", 0, 179, 25, help="Try 20-30 for Brown/Red onions.")
    s_min = st.slider("Saturation Min", 0, 255, 40, help="Increase to remove white table.")
    v_min = st.slider("Brightness (Value) Min", 0, 255, 40, help="Increase to ignore shadows.")

    st.divider()
    st.header("3. Shape Filters")
    min_area = st.number_input("Min Area (Size)", value=2000)
    circ_thresh = st.slider("Min Circularity", 0.0, 1.0, 0.65, step=0.05, 
                            help="1.0 = Perfect Circle. Lower for potatoes/irregular shapes.")
    
    st.divider()
    show_masks = st.checkbox("Show Debug Masks", value=False)

# --- PROCESSING ENGINE ---
def analyze_ultimate(uploaded_file, real_ref_mm, h_min, h_max, s_min, v_min, circ_thresh):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, "Error decoding image."
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- STEP A: FIND REFERENCE (BLUE) ---
    # We keep Blue fixed to keep it simple, as it works well.
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_ref = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Clean up Ref Mask
    kernel = np.ones((5,5), np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel)
    
    # Find Ref Contours
    cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_ref:
        return None, "Reference Not Found! Please place a BLUE object in the photo."
    
    # Get the biggest blue object
    ref_contour = max(cnts_ref, key=cv2.contourArea)
    ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
    px_per_mm = (ref_radius * 2) / real_ref_mm
    
    # --- STEP B: FIND ONIONS (USER SETTINGS) ---
    # Handle "Red Wrap-Around" Logic
    # If user sets Min=170 and Max=10, we need TWO masks combined.
    if h_min > h_max:
        # Range 1: h_min to 179
        lower1 = np.array([h_min, s_min, v_min])
        upper1 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Range 2: 0 to h_max
        lower2 = np.array([0, s_min, v_min])
        upper2 = np.array([h_max, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask_onion = cv2.add(mask1, mask2)
    else:
        # Standard Range (e.g., 10 to 30)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, 255, 255])
        mask_onion = cv2.inRange(hsv, lower, upper)
    
    # Clean up Onion Mask
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel, iterations=2) # Remove noise
    mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel, iterations=2) # Close holes
    
    # We subtract the Reference Mask from the Onion Mask
    # This prevents the Blue Cap from being detected as an onion if colors overlap
    mask_onion = cv2.subtract(mask_onion, mask_ref)
    
    cnts_onion, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_onions = []
    skipped_onions = [] # For debugging
    
    for c in cnts_onion:
        area = cv2.contourArea(c)
        if area > min_area:
            # CIRCULARITY CHECK
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            
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
    result = analyze_ultimate(uploaded_file, ref_real_size, h_min, h_max, s_min, v_min, circ_thresh)
    
    if result and len(result) == 4:
        sizes, final_img, mask_o, mask_r = result
        
        st.image(final_img, channels="BGR", caption="Blue=Ref | Green=Valid Onion | Red=Non-Circular", use_container_width=True)
        
        if show_masks:
            c1, c2 = st.columns(2)
            c1.image(mask_r, caption="Blue Mask (Ref)", use_container_width=True)
            c2.image(mask_o, caption="Onion Mask (Based on Sliders)", use_container_width=True)
            
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
