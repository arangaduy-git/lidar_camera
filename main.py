import cv2
import math
from ultralytics import YOLO
import numpy as np
import sys
import time
import signal

from rplidar import RpLidar
import open3d as o3d


def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer",
                      width=800, height=800)    
    render_opt = vis.get_render_option()
    render_opt.point_size = 5.0        
    render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    pc = o3d.geometry.PointCloud()
    vis.add_geometry(pc)
    zoom_coefficient = 1.39
    camera_matrix = np.array( [[698.58832204 ,  0. ,        269.94691245],
    [  0.,         677.51122202, 197.87670955],
    [  0.,           0.,           1.        ]])

    dist_coefs = np.array([-0.62594526,  0.54754649,  0.02941673,  0.02271911, -0.42414422])

    lidar = RpLidar()
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    xyxy_opr = [40, 67, 564, 423]
    model = YOLO('yolov8s.pt')


    def draw(points, colors):        
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pc)
        ret = vis.poll_events()
        if ret:
            vis.update_renderer() 
        return ret  


    def finalize():
        vis.destroy_window()        
        lidar.stop()
        cam.release()
        cv2.destroyAllWindows()
        

    def signal_handler(sig, frame):
        finalize()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    

    def crop_image(img):
        x1 = xyxy_opr[0]
        y1 = xyxy_opr[1]
        x2 = xyxy_opr[2]
        y2 = xyxy_opr[3]

        img_cropped = img[y1:y2,x1:x2]
        return img_cropped


    def scan_to_numpy_arrays(scan, boxes):
        data = []
        colors = []

        for x in range(250):
            data.append([x / 100, x / 100, 0])
            colors.append([1, 0, 1])
            data.append([x / 100, -x / 100, 0])
            colors.append([1, 0, 1])

        for x in range(250):
            k1 = -(frame_width / 2 - xyxy_opr[0]) / frame_width * 2
            k2 = (xyxy_opr[2] - frame_width / 2) / frame_width * 2
            data.append([x / 100, x / 100 * k1, 0])
            colors.append([1, 1, 0])
            data.append([x / 100, x / 100 * k2, 0])
            colors.append([1, 1, 0])

        for i in range(0, int(len(scan)), 3):
            point = [scan[i], scan[i+1], scan[i+2]]  
            if point != [0, 0, 0] and point != [0, -0, 0]:
                data.append(point)
                if scan[i + 1] < scan[i] and -scan[i + 1] < scan[i] and boxes:
                    f = False
                    for k in range(len(boxes)):
                        min_box = boxes[k][0] / 100 / (frame_width / 200 / scan[i]) - scan[i]
                        max_box = boxes[k][1] / 100 / (frame_width / 200 / scan[i]) - scan[i]
                        if min_box <= scan[i+1] <= max_box:
                            f = True
                            break
                    if f:
                        colors.append([1, 0, 0])
                    else:
                        colors.append([0, 1, 0])
                else:
                    colors.append([0, 1, 0])

        if boxes:
            for h in range(len(boxes)):
                for c in range(1, 250):
                    v = c / 100
                    data.append([v, int(boxes[h][1]) / 100 / (frame_width / 200 / v) - v, 0])
                    colors.append([1, 1, 1])
                    data.append([v, int(boxes[h][0]) / 100 / (frame_width / 200 / v) - v, 0])
                    colors.append([1, 1, 1])
        return np.array(data), np.array(colors)


    while True:
        ret, frame = cam.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)[:,::-1]
        frame = cv2.flip(frame, 1).copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        frame = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dst = crop_image(frame)

        results = model(dst, verbose=False)[0]
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        peoples_boxes = []
        if 0 in classes:
            for class_id, box in zip(classes, boxes):
                if class_id == 0:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1 + xyxy_opr[0], y1 + xyxy_opr[1]), (x2 + xyxy_opr[0], y2 + xyxy_opr[1]), (255, 0, 0), 2)
                    cv2.putText(frame, 'person', (x1 + 5 + xyxy_opr[0], y1 - 5 + xyxy_opr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    peoples_boxes.append((x1 + xyxy_opr[0] - 5, x2 + xyxy_opr[0] + 5))
        
        cv2.rectangle(frame, (xyxy_opr[0], xyxy_opr[1]), (xyxy_opr[2], xyxy_opr[3]), (255, 0, 255), 4)

        scan = lidar.get_scan()
        if len(scan) == 0:  
            time.sleep(0.01)                 
            continue
  
        points, colors = scan_to_numpy_arrays(scan, peoples_boxes)

        if not draw(points, colors):
           break

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    finalize()


if __name__ == "__main__":
    main()
