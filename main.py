import cv2
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

    lidar = RpLidar()
    cam = cv2.VideoCapture(0)
    model = YOLO('yolov8s.pt')
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


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


    def scan_to_numpy_arrays(scan, boxes):
        data = []
        colors = []
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

        for i in range(250):
            data.append([i / 100, i / 100, 0])
            colors.append([1, 0, 1])
            data.append([i / 100, -i / 100, 0])
            colors.append([1, 0, 1])

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
        results = model(frame, verbose=False)[0]
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

        peoples_boxes = []
        if 0 in classes:
            for class_id, box in zip(classes, boxes):
                if class_id == 0:
                    x1, y1, x2, y2 = box
                    peoples_boxes.append((x1 - 10, x2 + 10))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, 'person', (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
