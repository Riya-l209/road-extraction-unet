from flask import Flask, request, render_template, send_file
from deployment.utils import load_model, preprocess_image, postprocess_mask, save_mask
import os
import torch

app = Flask(__name__)
model = load_model()  # Load model once at startup

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = 'static/input.png'
        image_file.save(image_path)

        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = model(image_tensor)

        mask = postprocess_mask(output)
        mask_path = save_mask(mask)

        return send_file(mask_path, mimetype='image/png')

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
