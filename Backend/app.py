import os
import base64
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


MODEL_PATH = 'waste_classifier_best.h5'
waste_classifier = load_model(MODEL_PATH)


CLASS_LABELS = ['biodegradable', 'non-biodegradable']
IMG_SIZE = 224  

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    try:
      
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
       
        img_array = img_to_array(img)
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0) 
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def classify_waste(image_path):
    """Classify the waste image using the trained model"""
    try:
       
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return {"success": False, "error": "Image preprocessing failed"}
        
       
        prediction = waste_classifier.predict(processed_img)
        
       
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        return {
            "success": True,
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error classifying waste: {str(e)}")
        return {"success": False, "error": f"Classification failed: {str(e)}"}

def analyze_image_with_gemini(image_path):
    try:
        client = OpenAI(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a waste classification expert who provides impactful, fact-based responses with specific statistics and clear disposal instructions. Be direct and use an urgent, compelling tone."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Identify the specific type of waste in this image. Then provide a catchy, impactful 1 line response that includes: (1) A striking statistic about this waste type's environmental impact (e.g., 'Plastic contributes to 40% of ocean pollution'), and (2) Clear, direct instructions for proper disposal. Make your response memorable and motivating."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]}
            ]
        )
        print("Gemini Response:", response)
        return {
            "success": True,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        print(f"Exception during Gemini API call: {str(e)}")
        return {
            "success": False,
            "error": f"API call failed: {str(e)}"
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        print("Data Received:", data)

        if not data or 'image' not in data or 'filename' not in data:
            return jsonify({'error': 'Missing image data or filename'}), 400

        base64_image = data['image']
        filename = data['filename']

       
        base64_data = re.sub('^data:image/.+;base64,', '', base64_image)
        image_data = base64.b64decode(base64_data)

      
        unique_filename = f"{os.path.splitext(filename)[0]}_{os.urandom(4).hex()}{os.path.splitext(filename)[1]}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

       
        with open(file_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved to {file_path}")

      
        classification_result = classify_waste(file_path)
        
        
        gemini_result = analyze_image_with_gemini(file_path)
        
       
        if classification_result["success"] and gemini_result["success"]:
            combined_result = {
                "success": True,
                "model_classification": {
                    "class": classification_result["class"],
                    "confidence": classification_result["confidence"]
                },
                "gemini_analysis": gemini_result["response"]
            }
            return jsonify(combined_result), 200
        else:
            errors = []
            if not classification_result["success"]:
                errors.append(classification_result.get("error", "Model classification failed"))
            if not gemini_result["success"]:
                errors.append(gemini_result.get("error", "Gemini analysis failed"))
            
            return jsonify({
                "success": False,
                "errors": errors
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)