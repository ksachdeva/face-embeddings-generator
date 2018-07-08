import dlib
import os
import json

def generate_embeddings(dataset, models_dir, out_dir):
    face_rec_model_path = os.path.join(os.path.abspath(models_dir), 'dlib_face_recognition_resnet_model_v1.dat')
    shape_model_path = os.path.join(os.path.abspath(models_dir), 'shape_predictor_5_face_landmarks.dat')

    # build detector, predictor and recognizer
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(shape_model_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    data = []

    # go over each class and its images in the dataset
    # and compute the encodings
    for e in dataset:
        print('Processing %s ..' % e.name)
        embeddings = []
        for f in e.image_paths:
            img = dlib.load_rgb_image(f)
            dets = detector(img, 1)
            # Only process images that have one face
            if len(dets) != 1:
                continue

            for d in dets:
                shape = sp(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                embeddings.append(list(face_descriptor))

        anEntry = {'name' : e.name, 'embeddings' : embeddings}
        data.append(anEntry)

    out_file_path = os.path.join(os.path.abspath(out_dir), 'dlib.json')
    with open(out_file_path, 'w+') as outfile:
        json.dump(data, outfile)

