import numpy as np
import cv2


def zoom_center(img, zoom_factor=1.5):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


camera_matrix = np.array( [[698.58832204 ,  0. ,        269.94691245],
 [  0.,         677.51122202, 197.87670955],
 [  0.,           0.,           1.        ]])

dist_coefs = np.array([-0.62594526,  0.54754649,  0.02941673,  0.02271911, -0.42414422])

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)[:,::-1]
    frame = cv2.flip(frame, 1).copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    cv2.imshow('Camera', zoom_center(dst, 1.39))

    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()