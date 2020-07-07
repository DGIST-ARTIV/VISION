import cv2

capture = cv2.VideoCapture("test.webm")

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    print(frame.shape)
    if cv2.waitKey(33) > 0: break

capture.release()
cv2.destroyAllWindows()
