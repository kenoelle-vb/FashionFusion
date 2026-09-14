import cv2
import cvzone
import os 

path = "C:/Users/keno/OneDrive/Pictures/Camera Roll/testforsgu2.mp4"
cam = 0

cap = cv2.VideoCapture(cam)

shirtFolderPath = "C:/Users/keno/OneDrive/Documents/SGU Hackathon/PNG Version"
listShirts = os.listdir(shirtFolderPath)

imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[1]), cv2.IMREAD_UNCHANGED)
imgShirt = cv2.resize(imgShirt,(0,0), None, 0.75, 0.75)

imgPants = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)
imgPants = cv2.resize(imgPants,(0,0), None, 0.9, 0.9)

while True:
    success, img = cap.read()
    try:
        cvzone.overlayPNG(img, imgShirt, (200, 50))
        cvzone.overlayPNG(img, imgPants, (180, 250))
    except:
        pass
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)