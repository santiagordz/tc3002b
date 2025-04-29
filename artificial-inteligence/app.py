from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = "results/cnn_detector_20250428-133221_final_best.keras"  # Update with your actual model path
CLASS_NAMES = ["clubs", "diamonds", "hearts", "spades"]
IMG_SIZE = 224  # Should match what your model was trained on

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def preprocess_image(image_bytes):
    """Process the image bytes to make it suitable for the model"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')  # Ensure image is RGB
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Return prediction
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create a simple HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Playing Card Suit Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-container {
            margin-top: 20px;
            display: none;
        }
        .upload-form {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #preview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px 0;
            display: none;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .prediction {
            font-weight: bold;
            font-size: 24px;
            color: #2c3e50;
        }
        .confidence {
            color: #7f8c8d;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #all-probabilities {
            margin-top: 20px;
        }
        .probability-bar {
            height: 20px;
            background-color: #3498db;
            margin-bottom: 5px;
            border-radius: 3px;
        }
        .probability-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }
    </style>
</head>
<body>
    <h1>Playing Card Suit Detector</h1>
    <div class="container">
        <div class="upload-form">
            <p>Upload an image of a playing card to determine its suit.</p>
            <input type="file" id="imageUpload" accept="image/*">
            <img id="preview" src="#" alt="Image preview">
            <button id="predictBtn">Predict Suit</button>
        </div>
        <div class="loader" id="loader"></div>
        <div class="result-container" id="resultContainer">
            <h2>Result:</h2>
            <p>The card is: <span class="prediction" id="prediction"></span></p>
            <p class="confidence">Confidence: <span id="confidence"></span>%</p>
            
            <div id="all-probabilities">
                <h3>All Probabilities:</h3>
                <div id="probabilityBars"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('imageUpload').addEventListener('change', function(e) {
            const preview = document.getElementById('preview');
            preview.src = URL.createObjectURL(e.target.files[0]);
            preview.style.display = 'block';
        });

        document.getElementById('predictBtn').addEventListener('click', function() {
            const fileInput = document.getElementById('imageUpload');
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            // Show loader, hide results
            document.getElementById('loader').style.display = 'block';
            document.getElementById('resultContainer').style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Display results
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2);
                document.getElementById('resultContainer').style.display = 'block';

                // Display probability bars
                const barsContainer = document.getElementById('probabilityBars');
                barsContainer.innerHTML = '';
                
                Object.entries(data.probabilities).forEach(([suit, prob]) => {
                    const percentage = (prob * 100).toFixed(2);
                    
                    // Create label
                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'probability-label';
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.textContent = suit;
                    
                    const valueSpan = document.createElement('span');
                    valueSpan.textContent = `${percentage}%`;
                    
                    labelDiv.appendChild(nameSpan);
                    labelDiv.appendChild(valueSpan);
                    barsContainer.appendChild(labelDiv);
                    
                    // Create bar
                    const barDiv = document.createElement('div');
                    barDiv.className = 'probability-bar';
                    barDiv.style.width = `${percentage}%`;
                    barsContainer.appendChild(barDiv);
                });
            })
            .catch(error => {
                document.getElementById('loader').style.display = 'none';
                alert('Error: ' + error);
            });
        });
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)