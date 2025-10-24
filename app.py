from flask import Flask, render_template, request, Response
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

print("+"*50, "loading model")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)

# Initialize webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Webcam Feed Page
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    info['language'] = request.form['language']
    print(info)
    return render_template('choose_singer.html', data=info['language'])

@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
    info['singer'] = request.form['singer']

    found = False
    while not found:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y+h, x:x+w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))
    roi = roi / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)
    prediction = np.argmax(prediction)
    detected_emotion = label_map[prediction]  # Get the emotion label

    link = f"https://www.youtube.com/results?search_query={info['singer']}+{detected_emotion}+{info['language']}+song"

    return render_template("emotion_detect.html", emotion=detected_emotion, link=link)

if __name__ == "__main__":
    app.run(debug=True)
