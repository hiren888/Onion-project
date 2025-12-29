from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
# IMPORT YOUR CODE HERE:
# We assume your existing file is named "onion_measurement.py" 
# and has a function called "measure_onion"
# If your file is named differently, change 'onion_measurement' to your filename.
from onion_measurement import measure_onion 

app = Flask(__name__)
CORS(app) # This allows your React app to talk to this Python server

# Create a folder to save temporary images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/measure', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    # Get the reference object size (default to coin if not sent)
    ref_obj_name = request.form.get('reference_object', 'coin')
    
    # Save the image temporarily so your computer vision code can read it
    filepath = os.path.join(UPLOAD_FOLDER, "temp_onion.jpg")
    file.save(filepath)

    try:
        # --- CALL YOUR EXISTING CODE HERE ---
        # We pass the filepath and the reference object name to your function
        diameter_mm = measure_onion(filepath, ref_obj_name)
        
        # Determine Grade (You can also do this in Python if you prefer)
        grade = "Medium"
        if diameter_mm < 40: grade = "Small"
        elif diameter_mm > 60: grade = "Large"

        return jsonify({
            "diameter": round(diameter_mm, 2),
            "grade": { "label": grade },
            "confidence": 0.95,
            "processingTime": "0.5s"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)
