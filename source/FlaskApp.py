from flask import Flask, request
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
from matplotlib import pyplot as plt

app = Flask(__name__)
run_with_ngrok(app)


@app.route("/")
def hello():
    return "Hello Jevin World!"


@app.route('/recognize', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file'].read()

        ee = np.frombuffer(f, np.uint8)
        img = cv2.imdecode(ee, cv2.IMREAD_UNCHANGED)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run()
