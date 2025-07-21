import base64

from flask import Flask, request, jsonify
import cv2
import numpy as np

from ai import Processor
from logger import setup_logger

app = Flask(__name__)

shared_data = {}

logger = setup_logger('SmartGV Server')


@app.route('/upload', methods=['POST'])
def upload():
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

        shared_data['latest_img'] = img # TODO: share user and image, so you know whom you are sending what
        logger.debug('Shared image between endpoints')

        return jsonify({'status': 'Image displayed successfully'})

    logger.error('Invalid file type')
    return jsonify({'error': 'Invalid file type'}), 400


@app.route("/info", methods=["GET"])
def info():
    logger.debug('Extracting shared image')
    img = shared_data.get("latest_img")
    if img is None:
        logger.error("No image uploaded yet")
        return jsonify({"error": "No image uploaded yet"}), 400

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


# TODO: fix bug where if you send a random image it shows you the data for the latest image that the model could handle
# TODO: fix if you load one image after another it will show the first image unless you load the second image twice
