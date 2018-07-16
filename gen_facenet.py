from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import json
from tensorflow.python.ops import data_flow_ops
import time

def generate_embeddings(dataset,model_dir,out_dir):
    image_size = 160
    lfw_batch_size = 1#100
    use_flipped_images = 0
    use_fixed_image_standardization = 1
    paths = []
    for i in range(len(dataset)):
            nrof_image_1class = len(dataset[i])   
            for j in range(nrof_image_1class):
                paths.append(dataset[i].image_paths[j])

    with tf.Graph().as_default():   
        with tf.Session() as sess:

            # Get the paths for the corresponding images
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    
            nrof_preprocess_threads = 4
            image_size = (image_size, image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
    
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(model_dir, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")              
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            
            # Enqueue one epoch of image paths and labels
            nrof_embeddings = len(paths)
            nrof_flips = 2 if use_flipped_images else 1
            nrof_images = nrof_embeddings * nrof_flips
            labels_array = np.expand_dims(np.arange(0,nrof_images),1)
            image_paths_array = np.expand_dims(np.repeat(np.array(paths),nrof_flips),1)
            control_array = np.zeros_like(labels_array, np.int32)
            if use_fixed_image_standardization:
                control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
            if use_flipped_images:
                # Flip every second image
                control_array += (labels_array % 2)*facenet.FLIP
            sess.run(eval_enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
            
            embedding_size = int(embeddings.get_shape()[1])
            assert nrof_images % lfw_batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
            nrof_batches = nrof_images // lfw_batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            lab_array = np.zeros((nrof_images,))
            for i in range(nrof_batches):
                if i==1:
                    start_time = time.time()
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:lfw_batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab, :] = emb
                if i % 50 == 1:
                    print('.', end='')
                    sys.stdout.flush()        
            print('')   
            print("---everage inference time is %1.4f seconds per image---" % ((time.time() - start_time)/(nrof_batches-1)))
            
            embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
            if use_flipped_images:
                # Concatenate embeddings for flipped and non flipped version of the images
                embeddings[:,:embedding_size] = emb_array[0::2,:]
                embeddings[:,embedding_size:] = emb_array[1::2,:]
            else:
                embeddings = emb_array

    # Save embeddings to the json file
    data = []
    count = 0
    for e in dataset:
        print('Processing %s ..' % e.name)
        temp_embedding = []
        for _ in range(len(e)):
            temp_embedding.append(list(embeddings[count]))
            count+=1    
        anEntry = {'name' : e.name, 'embeddings' : temp_embedding}
        data.append(anEntry)
        
    out_file_path = os.path.join(os.path.abspath(out_dir), 'facenet.json')
    
    with open(out_file_path, 'w+') as outfile:
        json.dump(data, outfile)