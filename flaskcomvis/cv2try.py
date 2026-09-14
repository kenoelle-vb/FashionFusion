from flask import Flask, render_template, request, Response
import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
import numpy as np

app = Flask(__name__, template_folder='../flaskcomvis')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

shirtFolderPath = "C:/Users/keno/OneDrive/Documents/SGU Hackathon/PNG Version"
listShirts = os.listdir(shirtFolderPath)

detector = PoseDetector()

@app.route('/')
def index():
    return render_template('cv2try.htm')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    path = "C:/Users/keno/OneDrive/Pictures/Camera Roll/testforsgu2.mp4"
    cam = 0
    cap = cv2.VideoCapture(cam)
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            center_shoulder_x = (lm11[0] + lm12[0]) // 2
            center_shoulder_y = (lm11[1] + lm12[1]) // 2
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[1]), cv2.IMREAD_UNCHANGED) 
            imgShirt = cv2.resize(imgShirt,(0,0), None, 1, 1)
            imgPants = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED) 
            imgPants = cv2.resize(imgPants,(0,0), None, 1.25, 1.25)
            try : 
                cvzone.overlayPNG(img, imgShirt, (center_shoulder_x + 200, center_shoulder_y+ 385))
                cvzone.overlayPNG(img, imgPants, (center_shoulder_x + 160, center_shoulder_y+ 720))
            except : 
                pass
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return render_template('cv2try.htm')

if __name__ == '__main__':
    app.run(debug=True)