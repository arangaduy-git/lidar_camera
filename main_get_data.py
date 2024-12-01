import cv2
from ultralytics import YOLO
import numpy as np
import sys
import time
import signal

from rplidar import RpLidar

from multiprocessing.shared_memory import SharedMemory
import struct


def main():
    threshold_clusters = 0.2
    camera_matrix = np.array( [[698.58832204 ,  0. ,        269.94691245],
    [  0.,         677.51122202, 197.87670955],
    [  0.,           0.,           1.        ]])

    dist_coefs = np.array([-0.62594526,  0.54754649,  0.02941673,  0.02271911, -0.42414422])

    lidar = RpLidar()
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    xyxy_opr = [40, 67, 564, 423]
    model = YOLO('yolov8s.pt')
    shared_dots = SharedMemory(name='lidar_dots', size=8128, create=True)
    shared_boxes = SharedMemory(name='lidar_boxes', size=1024, create=True)
    new_frame_time = 0
    prev_frame_time = 0


    def finalize():     
        lidar.stop()
        cam.release()
        shared_dots.close()
        shared_boxes.close()
        cv2.destroyAllWindows()
        

    def signal_handler(sig, frame):
        finalize()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    

    def crop_image(img):
        # получаю изображение без черных линий
        x1 = xyxy_opr[0]
        y1 = xyxy_opr[1]
        x2 = xyxy_opr[2]
        y2 = xyxy_opr[3]

        img_cropped = img[y1:y2,x1:x2]
        return img_cropped
    

    def find_clusters(data, threshold_clusters=0.1):
        # Преобразуем данные в numpy массив
        data = np.array(data)
        
        # Функция для поиска ближайших соседей
        def is_close(point1, point2, threshold_clusters):
            return np.linalg.norm(point1 - point2) <= threshold_clusters

        # Алгоритм кластеризации
        clusters = []
        visited = set()
        for i, point in enumerate(data):
            if i in visited:
                continue
            
            # Создаём новый кластер
            cluster = []
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                
                visited.add(idx)
                cluster.append([data[idx][0], data[idx][1]])
                
                # Ищем соседей
                for j, other_point in enumerate(data):
                    if j not in visited and is_close(np.array([data[idx][0], data[idx][1]]), np.array([other_point[0], other_point[1]]), threshold_clusters):
                        queue.append(j)
            
            if cluster:
                clusters.append(np.array(cluster))
        
        return clusters
    

    def bounding_boxes(clusters):
        # Определяем прямоугольники для кластеров
        boxes = []
        for cluster in clusters:
            x_min, y_min = np.min(cluster, axis=0)
            x_max, y_max = np.max(cluster, axis=0)
            boxes.append(((x_min, y_min), (x_max, y_max)))
        return boxes

    def scan_to_numpy_arrays(scan, boxes):
        data_in_cam = []
        data_element_buf = 1
        counter_buf = 0

        # обработка основных точек
        for i in range(0, int(len(scan)), 3):
            point = [scan[i], scan[i+1], scan[i+2]]  
            if point != [0, 0, 0] and point != [0, -0, 0]:
                if scan[i + 1] < scan[i] and -scan[i + 1] < scan[i] and boxes:
                    f = False
                    counter_buf += 1
                    shared_dots.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', scan[i])
                    data_element_buf += 1
                    shared_dots.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', scan[i+1])
                    data_element_buf += 1
                    for k in range(len(boxes)):
                        min_box = boxes[k][0] / 100 / (frame_width / 200 / scan[i]) - scan[i]
                        max_box = boxes[k][1] / 100 / (frame_width / 200 / scan[i]) - scan[i]
                        if min_box <= scan[i+1] <= max_box:
                            f = True
                            break
                    if f:
                        # shared memory
                        data_in_cam.append(point)  # для поиска кластеров
        shared_dots.buf[0:8] = struct.pack('d', counter_buf)

        # обработка соседних точек
        clusters = find_clusters(data_in_cam, threshold_clusters)
        boxes_clusters = bounding_boxes(clusters)

        data_element_buf = 1
        counter_buf = 0

        # отрисовка прямоугольников
        for box in boxes_clusters:
            shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', box[0][0])
            data_element_buf += 1
            shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', box[0][1])
            data_element_buf += 1
            shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', box[1][0])
            data_element_buf += 1
            shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf] = struct.pack('d', box[1][1])
            data_element_buf += 1
            counter_buf += 1
        shared_boxes.buf[0:8] = struct.pack('d', counter_buf)


    while True:
        ret, frame = cam.read()
        new_frame_time = time.time()
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
  
        scan_to_numpy_arrays(scan, peoples_boxes)

        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 

        fps = 'fps: ' + str(int(fps)) 
        cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    finalize()



main()
