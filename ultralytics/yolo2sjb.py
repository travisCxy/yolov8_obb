# -*-coding:utf-8-*-
import os
import json
import cv2
import imagesize
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import math
import pdb

def three_points_to_lefttop_wh_theta(three_points, to_int):
    x1, y1, x2, y2, x3, y3 = three_points
    w = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    h = math.sqrt(math.pow(x2 - x3, 2) + math.pow(y2 - y3, 2))
    theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
    ret = (x1, y1, w, h, theta)

    # if theta<-45:
    #     print(theta)
    #     #pdb.set_trace()
    # if theta>45:
    #     print(theta)
    #     pdb.set_trace()


    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)


def three_points_to_lefttop_rightbottom_theta(three_points, to_int):
    x1, y1, w, h, theta = three_points_to_lefttop_wh_theta(three_points, False)

    x2 = x1 + w
    y2 = y1 + h

    ret = (x1, y1, x2, y2, theta)
    if not to_int:
        return ret
    else:
        return map(lambda x: int(round(x)), ret)

def get_exif_data(image):
    """获取图片的EXIF数据"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data

def get_orientation(exif_data):
    """从EXIF数据中获取图片的方向"""
    orientation_key = 'Orientation'
    if orientation_key in exif_data:
        return exif_data[orientation_key]
    return None

def adjust_for_rotation(image, orientation):
    """根据图片的旋转方向调整宽度和高度"""
    width, height = image.size
    if orientation == 3:
        # 180°旋转
        return width, height
    elif orientation == 6:
        # 90°旋转
        return height, width
    elif orientation == 8:
        # 270°旋转
        return height, width
    return width, height

def get_size(image_file):
    """获取图片的宽度和高度"""
    image = Image.open(image_file)
    exif_data = get_exif_data(image)
    orientation = get_orientation(exif_data)
    return adjust_for_rotation(image, orientation)


import math


def sort_rectangle_vertices(p1, p2, p3, p4):
    # 将所有点放入列表中
    points = [p1, p2, p3, p4]

    # 按照 x 坐标排序
    points.sort(key=lambda x: (x[0], x[1]))

    # 分离左边和右边的两个点
    left_points = points[:2]
    right_points = points[2:]

    #pdb.set_trace()
    # 对左右两边的点分别按 y 坐标排序
    left_points.sort(key=lambda x: x[1], reverse=False)
    right_points.sort(key=lambda x: x[1], reverse=False)

    # 组合排序后的点
    sorted_points = left_points + right_points

    return sorted_points


# 示例用法
# p1 = (1, 2)
# p2 = (5, 8)
# p3 = (1, 5)
# p4 = (5, 2)
#
# sorted_vertices = sort_rectangle_vertices(p1, p2, p3, p4)
# print("Sorted vertices:", sorted_vertices)
# #pdb.set_trace()




template = open("/mnt/server_data2/dataset/word_det/SJB_DET_20240701/images_with_labels/6.28/5dc022b3030e240a1b2b82e3.txt", 'r')
data = json.load(template)

#cls_map = {1:0, 2:1, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:0, 12:4, 13:5, 14:6, 15:7, 16:8, 17:0, 18:9}
cls_map = {0:1, 1:2, 2:3, 3:4, 4:12, 5:13, 6:14, 7:15, 8:16, 9:18}
#cls_map = {1:0, 2:1, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3, 10:0, 12:4, 13:5, 14:6, 15:7, 16:8, 17:0, 18:9}
input_dir = "/mnt/server_data2/dataset/word_det/SJB_DET_20240701/2024-02-27_local/"
#label_dir = "/mnt/server_data2/code/projects/object_detection/ultralytics/runs/obb/predict18/labels/"
label_dir = "/mnt/server_data2/code/projects/object_detection/ultralytics/runs/obb/predict23/labels/"
output_dir = "/mnt/server_data2/dataset/word_det/SJB_DET_20240701/2024-02-27_local/"
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(input_dir):
    if file.split(".")[-1] not in ["jpg", "png"]:
        continue
    #assert os.path.exists("/mnt/server_data2/dataset/word_det/SJB_DET_20240701/多学科/"+file)
    # if "1E06AA14-A24F-4404-BC61-777664F7E9D1_sm.jpg" not in file:
    #     continue
    width, height = get_size(input_dir+file)
    if  os.path.exists(label_dir+file.replace("jpg", "txt")) is not True:
        continue
    label_file = open(os.path.join(label_dir, file.replace("jpg", "txt")))
    label = label_file.readlines()
    cur_data = data.copy()
    new_regions = []
    for line in label:

        line = line.strip().split()
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[:9])
        conf = float(line[-1])
        x1 = min(max(x1, 0), 1)
        y1 = min(max(y1, 0), 1)
        x2 = min(max(x2, 0), 1)
        y2 = min(max(y2, 0), 1)
        x3 = min(max(x3, 0), 1)
        y3 = min(max(y3, 0), 1)
        x4 = min(max(x4, 0), 1)
        y4 = min(max(y4, 0), 1)
        x1 = x1 * width
        y1 = y1 * height
        x2 = x2 * width
        y2 = y2 * height
        x3 = x3 * width
        y3 = y3 * height
        x4 = x4 * width
        y4 = y4 * height
        lt, lb, rt, rb = sort_rectangle_vertices((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        x_left, y_left, x_right, y_right, theta = three_points_to_lefttop_rightbottom_theta([lt[0], lt[1],rt[0], rt[1],rb[0], rb[1]], True)
        if theta<-45 or theta>45:
            #print(x1, y1, x2, y2, x3, y3, x4, y4)
            print(file)
            continue
        cls = cls_map[int(cls)]
        new_regions.append({"cls": cls, "region": [int(x_left), int(y_left), int(x_right), int(y_right)], "rotation": theta, "result": [""], "conf": conf})
    cur_data["regions"] = new_regions
    print(output_dir+file.replace("jpg", "txt"))
    with open(output_dir+file.replace("jpg", "txt"), 'w') as f:
        f.write(json.dumps(cur_data, indent=4, ensure_ascii=False))