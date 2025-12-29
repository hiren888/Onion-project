from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import uuid
from datetime import datetime

# IMPORT YOUR CODE HERE:
# Ensure your measurement file is named "onion_measurement.py" 
# and has a function called "measure_onion" that returns the diameter in mm.
from onion_measurement import measure_onion 

app = Flask(__name__)
CORS(app) # Allows the Flutter app to talk to this server

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Rocket AI specifically asked for the endpoint "/detect"
@app.route('/detect', methods=['POST'])
def detect_onion():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # --- CALL YOUR EXISTING CODE ---
        # We assume your function returns a diameter (float)
        # If your code is complex, we might need to adjust this line.
        diameter_mm = measure_onion(filepath)
        
        # Calculate Grade
        grade_label = "Medium"
        if diameter_mm < 40: grade_label = "Small"
        elif diameter_mm > 60: grade_label = "Large"

        # --- FORMAT RESPONSE FOR ROCKET AI ---
        # Rocket AI expects: { "id": "...", "timestamp": "...", "onions": [...] }
        response_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "onions": [
                {
                    "label": grade_label,
                    "confidence": 0.95,
                    "diameter_mm": round(diameter_mm, 2),
                    # We add a fake bounding box [x, y, w, h] because the app expects it for drawing
                    # Since we don't know exactly where the onion is without updating your math code,
                    # we return a generic box or 0s.
                    "box": [100, 100, 200, 200] 
                }
            ]
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # We run on 0.0.0.0 to make it accessible to your phone/simulator
    app.run(debug=True, host='0.0.0.0', port=5000)
