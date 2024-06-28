# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import io
# from PIL import Image

# # Load your pre-trained model
# model = load_model('ml_model.h5')

# app = Flask(__name__)

# # Define the decoder
# decoder = {
#     0: 'Melanocytic nevi',
#     1: 'Melanoma',
#     2: 'Benign keratosis-like lesions',
#     3: 'Basal cell carcinoma',
#     4: 'Actinic keratoses',
#     5: 'Vascular lesions',
#     6: 'Dermatofibroma'
# }

# def preprocess_image(img):
#     """Preprocess the image to the required size and format."""
#     img = img.resize((224, 224))  # Resize to the target input size of the model
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0  # Normalize to [0, 1]
#     return img

# @app.route('/')
# def home():
#     """Default route to check deployment status."""
#     return 'Deployment is successful!'

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle the prediction request."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']

#     try:
#         # Read the image
#         img = Image.open(io.BytesIO(file.read()))
#         img = preprocess_image(img)
        
#         # Make the prediction
#         prediction = model.predict(img)
#         predicted_class_index = np.argmax(prediction, axis=1)[0]
#         predicted_class = decoder[predicted_class_index]

#         return jsonify({'prediction': predicted_class})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
from PIL import Image

# Path to the TensorFlow Lite model
tflite_model_path = 'model.tflite'

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

# Define the decoder
decoder = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

def preprocess_image(img):
    """Preprocess the image to the required size and format."""
    img = img.resize((224, 224))  # Resize to the target input size of the model
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to [0, 1]
    return img

@app.route('/')
def home():
    """Default route to check deployment status."""
    return 'Deployment is successful!'

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)
        
        # Set the tensor to point to the input data
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run the interpreter
        interpreter.invoke()

        # Extract the output data
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data, axis=1)[0]
        predicted_class = decoder[predicted_class_index]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
