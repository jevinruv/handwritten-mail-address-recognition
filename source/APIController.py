from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import cross_origin
import matplotlib.pyplot as plt

import cv2
import numpy as np
import json
import base64

from Service import Service

PREFIX = "/api"

app = Flask(__name__)
run_with_ngrok(app)

service = Service()


@app.route(PREFIX + "/")
@cross_origin(origin='*', headers=['access-control-allow-origin', 'Content-Type'])
def hello():
    return "Hello Jevin World!"


@app.route(PREFIX + '/recognize', methods=['GET', 'POST'])
@cross_origin(origin='*', headers=['access-control-allow-origin', 'Content-Type'])
def upload_file():
    if request.method == 'POST':

        content = request.json
        images = content['images']
        results = []

        for image in images:

            nparr = np.frombuffer(base64.b64decode(image), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            text = service.recognize_text(img, "api")

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

            img_obj = {}
            img_obj['image'] = image
            img_obj['text'] = text
            results.append(img_obj)


    return jsonify(results)

if __name__ == '__main__':
    app.run()
