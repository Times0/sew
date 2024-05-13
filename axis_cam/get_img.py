import cv2
import requests
import numpy as np
import keyboard
import os
import datetime

format = "%Y-%m-%d %H-%M-%S"
os.makedirs("images", exist_ok=True)

def main():
    # URL of the Axis camera image
    flux_url = "rtsp://root:sewusocome@192.168.15.56/axis-media/media.amp"    
    cap = cv2.VideoCapture(flux_url)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (width, height))
        if not ret:
            print("Error: Could not read frame")
            break
        
        to_show = cv2.resize(frame, (int(1920 * 3/4), int(1080 *3/4)))
        cv2.imshow("frame", to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if space pressed save the image with date and time
        if keyboard.is_pressed("space"):
            with open("images/" + datetime.datetime.now().strftime(format) + ".jpg", "wb") as file: 
                file.write(cv2.imencode(".jpg", frame)[1].tobytes())
                
            print("Image saved")


if __name__ == "__main__":
    main()
