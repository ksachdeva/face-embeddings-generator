import sys
import argparse
import lfw
import gen_dlib

def main(args):
    dataset = lfw.get_dataset(args.lfw_dir, args.max_num_classes, args.min_images_per_class)

    if args.network == 'dlib':
        gen_dlib.generate_embeddings(dataset, args.models_dir, args.out_dir)
    else:
        raise NotImplementedError('Facenet is not supported yet')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('network', type=str, help='Network to use', choices=['dlib', 'facenet'], default='dlib')
    parser.add_argument('--lfw-dir', type=str, help='Path to the LFW directory')
    parser.add_argument('--max-num-classes', type=int, help='Maximum number of classes', default=10)
    parser.add_argument('--min-images-per-class', type=int, help='Min number of images per class', default=10)
    parser.add_argument('--models-dir', type=str, help='Path to the models directory')
    parser.add_argument('--out-dir', type=str, help='Path to the directory in which to write the embedding file')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))