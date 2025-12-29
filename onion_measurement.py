import cv2
import numpy as np

def measure_onion(image_path, reference_width_mm=30.0):
    """
    Analyzes an image using Color Segmentation (Green Ref, Red/Brown Onion).
    Adapted from Streamlit logic for Backend Server.
    """
    try:
        # --- CONFIGURATION (Defaults from your Streamlit Sliders) ---
        # 1. Reference Settings (Green Cap)
        # Hue: 35-90, Sat: 80+, Val: 70+
        REF_H_MIN, REF_H_MAX = 35, 90
        REF_S_MIN, REF_V_MIN = 80, 70
        
        # 2. Onion Settings (Red/Brown)
        # Hue: 160-20 (Wraps around), Sat: 60+, Val: 50+
        ONION_H_MIN, ONION_H_MAX = 160, 20 
        ONION_S_MIN, ONION_V_MIN = 60, 50
        
        # 3. Filtering
        MIN_AREA_THRESH = 4000
        SPROUT_K_SIZE = 11

        # --- LOAD IMAGE ---
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image from path.")
            return 0.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- STEP A: FIND GREEN REFERENCE ---
        lower_green = np.array([REF_H_MIN, REF_S_MIN, REF_V_MIN])
        upper_green = np.array([REF_H_MAX, 255, 255])
        mask_ref = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up mask (Close holes, Erode noise)
        kernel_ref = np.ones((5,5), np.uint8)
        mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel_ref, iterations=2)
        mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_ERODE, kernel_ref, iterations=1)

        cnts_ref, _ = cv2.findContours(mask_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts_ref:
            print("Error: Green Reference Object not detected.")
            return 0.0

        # Calculate Pixels Per Millimeter using the largest green object
        ref_contour = max(cnts_ref, key=cv2.contourArea)
        ((_, _), ref_radius) = cv2.minEnclosingCircle(ref_contour)
        
        if ref_radius == 0: return 0.0
        
        # Diameter in pixels = radius * 2
        px_per_mm = (ref_radius * 2) / reference_width_mm

        # --- STEP B: FIND ONION (RED/BROWN) ---
        # Handle Hue Wrap-around (Red exists at 160-180 AND 0-20)
        if ONION_H_MIN > ONION_H_MAX:
            mask1 = cv2.inRange(hsv, np.array([ONION_H_MIN, ONION_S_MIN, ONION_V_MIN]), np.array([179, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, ONION_S_MIN, ONION_V_MIN]), np.array([ONION_H_MAX, 255, 255]))
            mask_onion = cv2.add(mask1, mask2)
        else:
            mask_onion = cv2.inRange(hsv, np.array([ONION_H_MIN, ONION_S_MIN, ONION_V_MIN]), np.array([ONION_H_MAX, 255, 255]))

        # Fill Holes inside the onion (so we get the full shape)
        contours_temp, _ = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(mask_onion)
        
        if contours_temp:
            # Only fill substantial contours
            big_contours = [c for c in contours_temp if cv2.contourArea(c) > (MIN_AREA_THRESH / 4)]
            cv2.drawContours(mask_filled, big_contours, -1, 255, thickness=cv2.FILLED)

        # Subtract the Green Reference from the Onion Mask (prevent overlap issues)
        mask_ref_dilated = cv2.dilate(mask_ref, np.ones((15,15), np.uint8), iterations=1)
        mask_final = cv2.subtract(mask_filled, mask_ref_dilated)

        # Sprout Removal (Morphological Opening)
        if SPROUT_K_SIZE > 1:
            kernel_sprout = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPROUT_K_SIZE, SPROUT_K_SIZE))
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel_sprout)

        # Final Smoothing
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

        # --- STEP C: CALCULATE SIZE ---
        cnts_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sizes = []
        for c in cnts_final:
            if cv2.contourArea(c) > MIN_AREA_THRESH:
                # Use Ellipse fitting (Matches your "Min Axis" logic for sprouts)
                if len(c) < 5: 
                    # Fallback for simple shapes
                    ((_, _), radius) = cv2.minEnclosingCircle(c)
                    dia_mm = (radius * 2) / px_per_mm
                else:
                    (center, (MA, ma), angle) = cv2.fitEllipse(c)
                    axes = sorted([MA, ma])
                    minor_axis = axes[0] # Width (ignores sprout length)
                    dia_mm = minor_axis / px_per_mm
                
                sizes.append(dia_mm)

        if not sizes:
            return 0.0

        # Return the largest onion found
        return max(sizes)

    except Exception as e:
        print(f"Logic Error: {e}")
        return 0.0
