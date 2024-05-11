from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_tumor(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    return prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_tumor', methods=['POST'])
def detect_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']
    file_path = 'uploaded_image.jpg'
    file.save(file_path)

    prediction = predict_tumor(file_path)

    if prediction > 0.5:
        result = "Tumor Detected"
    else:
        result = "No Tumor Detected"

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
