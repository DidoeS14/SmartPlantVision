import base64

from flask import Flask, request, jsonify
import cv2
import numpy as np

from ai import Processor
from logger import setup_logger

app = Flask(__name__)


logger = setup_logger('SmartPV Server')

@app.route('/analyze', methods=['POST','GET'])
def analyze():
    if 'file' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and file.mimetype.startswith('image/'):
        # Read image from file stream
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        logger.debug('Extracted image into cv2 processable format')

        if img is None:
            logger.error('Failed to decode image')
            return jsonify({'error': 'Failed to decode image'}), 400

        logger.debug('Loading the AI processor')
        processor = Processor()
        logger.debug('Providing image for the processor')
        processor.set_img(img)
        logger.debug('Running the processor')
        processor.run()
        logger.debug('Receiving data from processor')
        data = processor.get_data()

        # Encode the image to JPEG in memory
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            logger.error("Image encoding failed")
            return jsonify({"error": "Image encoding failed"}), 500

        # Convert buffer to base64
        logger.debug('Converting buffer to base64 image')
        encoded_string = base64.b64encode(buffer).decode("utf-8")

        logger.debug(f'Sending data to client :{data}')
        return jsonify({
            "data": data,
            "image_base64": encoded_string
        })


if __name__ == '__main__':
    app.run(debug=True, port=8000)
