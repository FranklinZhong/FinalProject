from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')

# 更新模型的路径
MODEL_PATHS = {
    'face': 'A:\FinalProject\GUI\humanFace.h5',
    'plant': 'A:\FinalProject\GUI\plantDisease.h5',
    'skin': 'A:\FinalProject\GUI\skinCancer.h5'
}

models = {name: tf.keras.models.load_model(path) for name, path in MODEL_PATHS.items()}

# 预处理参数配置
PREPROCESS_CONFIG = {
    'face': {'size': (100, 120), 'labels': ['female', 'male']},
    'skin': {'size': (224, 224), 'labels': ['benign', 'malignant']},
    'plant': {'size': (256, 256), 'labels': ['Apple___Apple_scab', 
                                            'Apple___Black_rot', 
                                            'Apple___Cedar_apple_rust', 
                                            'Apple___healthy', 
                                            'Blueberry___healthy', 
                                            'Cherry_(including_sour)___Powdery_mildew', 
                                            'Cherry_(including_sour)___healthy', 
                                            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                            'Corn_(maize)___Common_rust_', 
                                            'Corn_(maize)___Northern_Leaf_Blight', 
                                            'Corn_(maize)___healthy', 
                                            'Grape___Black_rot', 
                                            'Grape___Esca_(Black_Measles)', 
                                            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                            'Grape___healthy', 
                                            'Orange___Haunglongbing_(Citrus_greening)', 
                                            'Peach___Bacterial_spot', 
                                            'Peach___healthy', 
                                            'Pepper,_bell___Bacterial_spot', 
                                            'Pepper,_bell___healthy', 
                                            'Potato___Early_blight', 
                                            'Potato___Late_blight', 
                                            'Potato___healthy', 
                                            'Raspberry___healthy', 
                                            'Soybean___healthy', 
                                            'Squash___Powdery_mildew', 
                                            'Strawberry___Leaf_scorch', 
                                            'Strawberry___healthy', 
                                            'Tomato___Bacterial_spot', 
                                            'Tomato___Early_blight', 
                                            'Tomato___Late_blight', 
                                            'Tomato___Leaf_Mold', 
                                            'Tomato___Septoria_leaf_spot', 
                                            'Tomato___Spider_mites Two-spotted_spider_mite', 
                                            'Tomato___Target_Spot', 
                                            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                                            'Tomato___Tomato_mosaic_virus', 
                                            'Tomato___healthy']}
}

@app.route('/', methods=['GET'])
def index():
    return render_template('GUI.html')

STATIC_FOLDER = 'static\TestPic'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Retrieve the model choice from the form data
        model_choice = request.form['model_choice']
        
        # Secure the filename and create a path in the static folder to save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.static_folder, filename)  # Use the static_folder path
        file.save(filepath)  # Save the file to the filesystem

        # Process the image and predict
        image = preprocess_image(file, model_choice)
        if image is None:
            os.remove(filepath)  # If processing fails, remove the saved file
            return 'Error processing image'
        
        # Load the model and make predictions
        model = models[model_choice]
        predictions = model.predict(image)
        result = decode_predictions(predictions, model_choice)

        # Prepare the result to send to the template
        # Make sure filepath is just the filename as it's used in 'url_for'
        return render_template('result.html', image_path=filename, result=result)




def preprocess_image(file, model_choice):
    """根据选定的模型调整图片大小和预处理."""
    config = PREPROCESS_CONFIG[model_choice]
    image = Image.open(file.stream)
    image = image.resize(config['size'])
    image_array = np.expand_dims(np.array(image), axis=0)
    image_array = image_array / 255.0
    return image_array

def decode_predictions(predictions, model_choice):
    """根据选定的模型解码预测结果."""
    labels = PREPROCESS_CONFIG[model_choice]['labels']
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = labels[predicted_class_index]
    return {
        'category': predicted_class,
        'accuracy': f'{confidence:.2%}'
    }


if __name__ == '__main__':
    app.run(debug=True)
