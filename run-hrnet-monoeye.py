import cv2
from SimpleHRNet import SimpleHRNet
from misc import visualization
from matplotlib import pyplot as plt
import math
import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='gen1_7')
parser.add_argument('--imagedir', type=str, default='')
args = parser.parse_args()

DIR = args.file

# ROOT_DIR = os.path.abspath("./")
ROOT_DIR = args.imagedir
IMAGE_DIR = os.path.join(ROOT_DIR, DIR+'_mask')
OUT_DIR = os.path.join(ROOT_DIR, DIR+'_hrnet')

os.makedirs(OUT_DIR, exist_ok=True)

model = SimpleHRNet(48, 17, './weights/pose_hrnet_w48_384x288.pth', multiperson=False)
joints_dict = visualization.joints_dict()

file_names = next(os.walk(IMAGE_DIR))[2]

path_p = os.path.join(ROOT_DIR, 'hrnet_2d_pos.csv')
path_w = os.path.join(ROOT_DIR, DIR+'_pelvis.csv')

pos_text = ''
pelvis_text = ''

for file_name in sorted(file_names):
    image = cv2.imread(os.path.join(IMAGE_DIR, file_name), cv2.IMREAD_COLOR)
    joints = model.predict(image)
    
    pos2d = joints[0]
    
    for i in range(pos2d.shape[0]):
        pos_text += str(pos2d[i][0]) + ',' + str(pos2d[i][1])
        if i != pos2d.shape[0]-1:
            pos_text += ','
    
    pos_text += '\n'
    
    # y, xとなっているため入れ替える
    px = (pos2d[11][1] + pos2d[12][1])/2
    py = (pos2d[11][0] + pos2d[12][0])/2
    pelvis_text += str(px) + ',' + str(py) + '\n'
    
    image = visualization.draw_points_and_skeleton(image, joints[0], joints_dict['coco']['skeleton'], confidence_threshold=0.0)
    cv2.imwrite(os.path.join(OUT_DIR, file_name), image)
    
    print(file_name)
    
with open(path_p, mode='w') as f:
    f.write(pos_text)
    
with open(path_w, mode='w') as f:
    f.write(pelvis_text)
    
print('done.')