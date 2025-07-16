import base64

from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.mimetype.startswith('image/'):
        # Read image from file stream
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Show the image
        # cv2.imshow('Uploaded Image', img) # TODO: run detections with model and return info.py along with image data
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return jsonify({'status': 'Image displayed successfully'})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route("/info", methods=["GET"])
def info(): # TODO: make it so some token is required, so the server knows which image must be returned
    with open(r"C:\Users\deqnb\OneDrive\Картини\OculusScreenshot1655231333.jpeg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return jsonify({
        "text": "Here is your test image!",
        "image_base64": encoded_string
    })
if __name__ == '__main__':
    app.run(debug=True, port=8000)
