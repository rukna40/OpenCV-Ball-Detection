import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    # width=int(cap.get(3))
    # height=int(cap.get(4))
    #frame1=frame.copy()

    lower_gr=np.array([[[20, 60, 80]]])
    upper_gr=np.array([[[40, 255, 255]]])
    
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    blurred = cv2.medianBlur(blur, 15, 0)
    frame_hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)  
    mask_hsv = cv2.inRange(frame_hsv,lower_gr,upper_gr)
    mask_hsv = cv2.erode(mask_hsv, None, iterations=2)
    mask_hsv = cv2.dilate(mask_hsv, None, iterations=2)
    mask = cv2.GaussianBlur(mask_hsv, (15, 15), 0)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.15, 50 ,param1=50, param2=30,minRadius=5, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
    
    # cnts,_ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # if len(cnts) > 0:
    #     c = max(cnts, key=cv2.contourArea)
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     if M["m00"]!=0:
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    #     if radius > 10:
    #         cv2.circle(frame1, (int(x), int(y)), int(radius),(0, 255, 255), 2)
    #         cv2.circle(frame1, center, 5, (0, 0, 255), -1)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
