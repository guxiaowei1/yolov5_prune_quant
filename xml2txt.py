import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
 
 
 
def convert(size, box):
 
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
 
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
 
 
    return (x, y, w, h)
 
 
def convert_annotation(xml_files_path, save_txt_files_path, classes, classes_all):
    xml_ori_path = '/home/guxiaowei/yolov5-master/bozhou/xml/'
    xml_files = os.listdir(xml_files_path)
    for xml_name in xml_files:
        if not (xml_name.endswith('.jpg') or xml_name.endswith('.png') or xml_name.endswith('.JPG')):
            continue
        if not os.path.exists(os.path.join(xml_ori_path, xml_name[:-4]+'.xml')):
            continue
        
        print(xml_name)
        xml_file = os.path.join(xml_ori_path, xml_name[:-4] + '.xml')
        out_txt_path = os.path.join(save_txt_files_path, xml_name[:-4] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w == 0 or h == 0:
            imgtest = cv2.imread(os.path.join(xml_files_path, xml_name))
            w, h = imgtest.shape[1], imgtest.shape[0]

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                if cls not in classes_all:
                    raise Exception("found error object name! Name is:", cls, "in file:", xml_file)
                else:
                    continue
            cls_id = classes[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
if __name__ == "__main__":
   
    classes_all = {'JYZ': 0, 'GLKG': 1, 'ZSKG': 2, "TM": 3, "BYQ": 4, "DLSBLQ": 5, "DLSRDQ": 6, 'JYZK': 0,\
        'GANTA_QX': 1, 'GANTA_ZC': 2, "GANTOU_PS": 3, "GANTOU_ZC": 4, 'CTJYZ':0, 'BLZRDQ': 7, 'DD': 8}
    classes = {'JYZ': 0, 'GANTA_QX': 1, 'GANTA_ZC': 2, "GANTOU_PS": 3, "GANTOU_ZC": 4, 'JYZK': 0, 'jyz':0}
    # 1、voc格式的xml标签文件路径
    xml_files = '/home/guxiaowei/mmdetection-master/peiwang/valdata/'
    # 2、转化为yolo格式的txt标签文件存储路径
    save_txt_files = '/home/guxiaowei/yolov5-master/data/images/val/labels/'
 
    convert_annotation(xml_files, save_txt_files, classes, classes_all)
 