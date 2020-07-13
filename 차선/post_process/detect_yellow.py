import cv2
import cv2
import numpy as np
 
video_path = '/home/dgist/Desktop/data/원내주행영상/낮/0626/2020-06-26-173017.webm' 
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

while(True):
    # capture Frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize=(800,288))

    if ret:
        cv2.imshow('Video', frame)

        #px = frame[:144]

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # White-ish areas in image
        # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
        # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
        # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
        white_lower = np.array([np.round(  0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
        white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
        white_mask = cv2.inRange(hls, white_lower, white_upper)

        # Yellow-ish areas in image
        # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
        # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
        # S value must be above some threshold (we want at least some saturation), e.g. within [0.35 ... 1.0]
        yellow_lower = np.array([np.round( 30 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
        yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

        # Calculate combined mask, and masked image
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        masked = cv2.bitwise_and(frame, frame, mask = mask)
        
        cv2.imshow("white", white_mask)
        cv2.imshow("yellow",yellow_mask)
        cv2.imshow("mask", masked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


