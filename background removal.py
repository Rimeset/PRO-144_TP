# import cv2 to capture videofeed
import cv2
import numpy as np

camera = cv2.VideoCapture(0)

camera.set(3, 640)
camera.set(4, 480)

mountain = cv2.imread('mount everest.jpg')

mountain = cv2.resize(mountain, (640, 480))

while True:
    status, frame = camera.read()

    if status:
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_bound = np.array([0, 120, 50])
        upper_bound = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_bound, upper_bound)

        lower_bound = np.array([170, 120, 70])
        upper_bound = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound, upper_bound)

        mask = mask1 + mask2

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        mask_inv = cv2.bitwise_not(mask)

        foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

        background = cv2.bitwise_and(mountain, mountain, mask=mask)

        final_output = cv2.addWeighted(foreground, 1, background, 1, 0)

        cv2.imshow('frame', final_output)
        code = cv2.waitKey(1)
        if code == 32:  
            break
        
camera.release()
cv2.destroyAllWindows()
