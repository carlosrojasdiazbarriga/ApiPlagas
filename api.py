import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import io

# Initialize our Flask application and the TensorFlow model
app = Flask(__name__)
model = tf.saved_model.load('best_saved_model')  # Carga el modelo SavedModel

# Carga de nombres de clases y otras configuraciones
class_names = {0: 'Minador', 1: 'MoscaBlanca', 2: 'Oidio', 3: 'Pulgon', 4: 'sana', 5: 'roya'}

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the image data from the POST request
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file'})

    image_file = request.files['image']
    image_data = image_file.read()

    # Decode and convert image to NumPy array
    image = tf.image.decode_jpeg(image_data, channels=3)  # Assumes JPEG images
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    image = tf.image.resize(image, (640, 640))

    # Add extra dimension if needed (model input shape)
    if len(image.shape) < 4:
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions using the loaded TensorFlow model
    predictions = model(image)  # Obtener predicciones directamente

    confidence_threshold = 0.5  # Ajusta según sea necesario

    # Check if max prediction is above confidence threshold
    predicted_class_index = np.argmax(predictions[0])
    max_prediction = predictions[0][predicted_class_index]
    if max_prediction < confidence_threshold:
        return jsonify({'result': 'Imagen no clasificable o relevante'})

    # Create response with probabilities for all classes
    class_probabilities = {class_names[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}
    response = {'class_probabilities': class_probabilities}
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
