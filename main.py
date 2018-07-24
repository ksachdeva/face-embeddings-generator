import sys
import argparse
import lfw
import vggface2
import gen_dlib
import gen_facenet

def main(args):
    # choose to use either the lfw dataset or vggface2 dataset for testing
    if args.vggface2_json_dir is not '':
        dataset = vggface2.get_dataset(args.vggface2_json_dir, args.max_num_classes, args.min_images_per_class)
    else:
        dataset = lfw.get_dataset(args.lfw_dir, args.max_num_classes, args.min_images_per_class)
    # get face embeddings with either dlib or facenet
    if args.network == 'dlib':
        gen_dlib.generate_embeddings(dataset, args.models_dir, args.out_dir)
    elif args.network == 'facenet':
        gen_facenet.generate_embeddings(dataset, args.models_dir, args.out_dir)
    else:
        raise NotImplementedError('Please choose a supported model')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('network', type=str, help='Network to use', choices=['dlib', 'facenet'], default='dlib')
    parser.add_argument('--lfw-dir', type=str, help='Path to the LFW image directory')
    parser.add_argument('--vggface2-json-dir', type=str, help='Path to the VGGface2 json file', default = '')
    parser.add_argument('--max-num-classes', type=int, help='Maximum number of classes', default=10)
    parser.add_argument('--min-images-per-class', type=int, help='Min number of images per class', default=10)
    parser.add_argument('--models-dir', type=str, help='Path to the dlib or facenet models directory')
    parser.add_argument('--out-dir', type=str, help='Path to the directory in which to write the embedding file')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))