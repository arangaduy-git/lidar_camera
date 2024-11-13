import cv2

cam = cv2.VideoCapture(0)

i = 0
while True:
    ret, frame = cam.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)[:,::-1]
    frame = cv2.flip(frame, 1).copy()

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f'camera_distortion/images/{i}.png', frame)
        i += 1
        print('saved', i)

    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()