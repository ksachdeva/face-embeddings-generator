from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import align_dataset_mtcnn as al
import os
import sys
import json
from tensorflow.python.ops import data_flow_ops
import time

def generate_embeddings(dataset, model_dir, out_dir):
    # Serialize the path 
    paths = []
    for i in range(len(dataset)):
            nrof_image_1class = len(dataset[i])   
            for j in range(nrof_image_1class):
                paths.append(dataset[i].image_paths[j])
  
    with tf.Graph().as_default():  
        with tf.Session() as sess:
            # Load the facenet model
            print('Loading feature extraction model')
            facenet.load_model(model_dir)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            data = []
            count = 0
            for e in dataset:
                print('Processing %s ..' % e.name)
                emb_array = []           
                for image_path in e.image_paths:
                    im = al.load_rgb_image(image_path) # get rgb image form image path
                    if im is not None:
                        aligned_image = al.get_frontal_face_detector(im, do_prewhiten = True) # rescaled and aligned face image, with whitening as default
                        if aligned_image is not None:
                            aligned_image = np.expand_dims(aligned_image, axis = 0)
                            feed_dict = {images_placeholder:aligned_image, phase_train_placeholder:False }
                            face_descriptor = sess.run(embeddings, feed_dict=feed_dict) # get the embeddings
                            emb_array.append(face_descriptor.tolist()[0]) # append the embedding for each name                           
                            count += 1

                anEntry = {'name' : e.name, 'embeddings' : emb_array}
                data.append(anEntry)

    print("Skipped total number of %s images" %(nrof_images-count))
    out_file_path = os.path.join(os.path.abspath(out_dir), 'facenet.json')
    
    with open(out_file_path, 'w+') as outfile:
        json.dump(data, outfile)