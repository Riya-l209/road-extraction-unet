from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "✅ Flask app is running — U-Net Road Extraction Project"

# Route to handle image upload (extend this later with prediction)
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Save uploaded image to uploads folder
    upload_folder = 'deployment/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image.filename)
    image.save(image_path)

    # Placeholder response
    return jsonify({'message': 'Image uploaded successfully', 'path': image_path})

if __name__ == '__main__':
    app.run(debug=True)

