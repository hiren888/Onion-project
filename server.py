from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import sys

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow the Flutter app to talk to this server

# --- SAFE IMPORT BLOCK ---
# This block prevents the server from crashing immediately if 'onion_measurement.py' 
# is missing or has a syntax error.
onion_function = None
import_error_message = None

try:
    # Attempt to import the measure_onion function from your file
    from onion_measurement import measure_onion
    onion_function = measure_onion
    print("SUCCESS: Found 'onion_measurement.py' and loaded 'measure_onion'.")
except Exception as e:
    # If it fails, we record the error but keep the server running
    import_error_message = str(e)
    print(f"STARTUP ERROR: Could not load onion logic. Reason: {e}")
# -------------------------

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/detect', methods=['POST'])
def detect_onion():
    # 1. Check if the onion logic loaded correctly on startup
    if onion_function is None:
        return jsonify({
            "error": "Server Configuration Error",
            "details": f"The Python logic could not be loaded. Error: {import_error_message}",
            "hint": "Check Render logs or ensure 'onion_measurement.py' exists and has no syntax errors."
        }), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    # Generate unique filename to prevent overwrites
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # 2. Run your measurement logic
        # We pass the filepath to the function you created in onion_measurement.py
        diameter_mm = onion_function(filepath)
        
        # 3. Calculate Grade based on diameter
        # (You can adjust these thresholds based on your specific onion variety)
        grade_label = "Medium"
        if diameter_mm < 40: grade_label = "Small"
        elif diameter_mm > 60: grade_label = "Large"

        # 4. Format response for Rocket AI / Flutter
        # The structure matches what Rocket AI expects: { id, timestamp, onions: [] }
        return jsonify({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "onions": [{
                "label": grade_label,
                "confidence": 0.95,
                "diameter_mm": round(diameter_mm, 2),
                # We provide a default bounding box because the App expects it to draw a square
                "box": [100, 100, 200, 200]
            }]
        })

    except Exception as e:
        print(f"Processing Error: {e}")
        # Return a 500 error with the exception message
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # This endpoint helps you debug if the code is loaded without uploading an image
    status = "healthy" if onion_function else "degraded"
    return jsonify({
        "status": status, 
        "onion_logic_loaded": onion_function is not None,
        "startup_error": import_error_message
    }), 200

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible externally (required for Render and Mobile Apps)
    app.run(debug=True, host='0.0.0.0', port=5000)
