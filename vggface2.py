# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:52:14 2018

@author: Innovation Team
"""

import lfw
import json

def read_json_file(file_path):
    with open(file_path, 'r') as fr:
        return json.load(fr)

def get_dataset(path, max_num_classes, min_images_per_class,):  
    vggface2_data = read_json_file(path)
    data = []
    class_count = 0
    image_count = 0
    for item in vggface2_data:
        if class_count == max_num_classes:
            break
        if len(vggface2_data[item]) > min_images_per_class:
            data.append(lfw.ImageClass(item, vggface2_data[item]))
            class_count += 1
            image_count += len(vggface2_data[item])
    print("Selected a total number of %s images" % image_count)
    return data