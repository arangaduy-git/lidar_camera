import open3d as o3d
from multiprocessing.shared_memory import SharedMemory
import struct
import sys
import signal
import numpy as np
import decimal


shared_dots = SharedMemory(name='lidar_dots', create=False)
shared_boxes = SharedMemory(name='lidar_boxes', create=False)
xyxy_opr = [40, 67, 564, 423]
frame_width = 640

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud Visualizer",
                    width=800, height=800)    
render_opt = vis.get_render_option()
render_opt.point_size = 5.0        
render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
pc = o3d.geometry.PointCloud()
vis.add_geometry(pc)


def draw(points, colors):        
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(pc)
    ret = vis.poll_events()
    if ret:
        vis.update_renderer() 
    return ret 


def finalize():
    shared_dots.close()
    vis.destroy_window()


def drange(x, y, jump):
    # range() for float
    x = decimal.Decimal(x)
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)    


def signal_handler(sig, frame):
    finalize()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


while True:
    f = True
    points = []
    colors = []

    count_buf = int(struct.unpack('d', shared_dots.buf[0:8])[0])
    data_element_buf = 1
    for j in range(count_buf):
        data0 = shared_dots.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        data1 = shared_dots.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        points.append([struct.unpack('d', data0)[0], struct.unpack('d', data1)[0], 0])
        colors.append([255, 255, 255])


    count_buf = int(struct.unpack('d', shared_boxes.buf[0:8])[0])
    data_element_buf = 1
    for h in range(count_buf):
        data0 = shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        data1 = shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        data2 = shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        data3 = shared_boxes.buf[data_element_buf * 8 + data_element_buf:(data_element_buf + 1) * 8 + data_element_buf]
        data_element_buf += 1
        box = [[struct.unpack('d', data0)[0], struct.unpack('d', data1)[0]], [struct.unpack('d', data2)[0], struct.unpack('d', data3)[0]]]
        for j in drange(round(box[0][0], 2), round(box[1][0], 2), 0.01):
            points.append([j, box[0][1], 0])
            colors.append([0, 255, 255])
            points.append([j, box[1][1], 0])
            colors.append([0, 255, 255])
        for j in drange(round(box[0][1], 2), round(box[1][1], 2), 0.01):
            points.append([box[0][0], j, 0])
            colors.append([0, 255, 255])
            points.append([box[1][0], j, 0])
            colors.append([0, 255, 255])

    for x in range(250):
            k1 = -(frame_width / 2 - xyxy_opr[0]) / frame_width * 2
            k2 = (xyxy_opr[2] - frame_width / 2) / frame_width * 2
            points.append([x / 100, x / 100 * k1, 0])
            colors.append([1, 1, 0])
            points.append([x / 100, x / 100 * k2, 0])
            colors.append([1, 1, 0])

    if not draw(points, colors):
        break

finalize()
