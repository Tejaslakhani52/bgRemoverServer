# from flask import Flask, request, jsonify, send_file
# import cv2
# import numpy as np
# import io
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     image_data = request.files.get('image')
#     if not image_data:
#         return jsonify({'error': 'No image provided'}), 400
    
#     # Load the image and perform background removal
#     img = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
#     height, width = img.shape[:2]
#     img_bgr = img.copy()  # Make a copy in BGR format
    
#     # Find the contours of the main subject
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Calculate the bounding box that encompasses the main subject
#     x, y, w, h = cv2.boundingRect(contours[0])  # Assuming there's at least one contour
#     rect = (x, y, x + w, y + h)
    
#     # Adjust the bounding box to make sure it covers the subject well
#     rect = (rect[0] - 20, rect[1] - 20, rect[2] + 20, rect[3] + 20)
    
#     # Perform GrabCut using the adjusted bounding box
#     mask = np.zeros(img.shape[:2], np.uint8)
#     cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
#     # ... (rest of the processing)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
    



# from flask import Flask, request, jsonify, send_file
# import cv2
# import numpy as np
# import io
# from flask_cors import CORS

# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# app = Flask(__name__)
# CORS(app)

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     image_data = request.files.get('image')
#     if not image_data:
#         return jsonify({'error': 'No image provided'}), 400
    
#     # Load the image and perform background removal
#     img = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
#     height, width = img.shape[:2]
#     img = cv2.resize(img, (int(width * 0.3), int(height * 0.3)), interpolation=cv2.INTER_AREA)
#     img_bgr = img.copy()  # Make a copy in BGR format
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     mask = np.zeros(img.shape[:2], np.uint8)
#     rect = (10, 10, width - 30, height - 30)
#     cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     img1 = img * mask[:, :, np.newaxis]
#     background = img - img1
#     background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
#     final = background + img1
    
#     # Convert the processed image back to BGR before returning
#     processed_image_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    
#     # Convert the processed image to bytes
#     processed_image_bytes = cv2.imencode('.jpg', processed_image_bgr)[1].tobytes()
    
#     return send_file(io.BytesIO(processed_image_bytes), mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)  


 

from flask import Flask, request, jsonify, send_file
import io
from flask_cors import CORS
from PIL import Image
import rembg

app = Flask(__name__)
CORS(app, resources={r"/process_image": {"origins": "https://bg-remover-delta.vercel.app/"}})
 
@app.route('/process_image', methods=['POST'])
def process_image():
    image_data = request.files.get('image')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400
    
    # Load the input image
    img = Image.open(image_data)
    
    # Perform background removal using rembg
    output = rembg.remove(img)
    
    # Convert the processed image to bytes
    output_bytes = io.BytesIO()
    output.save(output_bytes, format='PNG')
    output_bytes.seek(0)
    
    return send_file(output_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
