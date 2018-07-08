# Face embeddings generator

This is a set of scripts to generate a json file containing the embeddings (descriptors) using various neural networks.

## Setup / Install instructions

```bash
# create a python virtual environment
virtualenv -p python3 .env3
# activate the environment
source .env3/bin/activate
```

```bash
# install the required libraries
pip install -r requirements.txt
```

## Download LFW

You can download Labeled Faces in the Wild (LFW) database from here -
[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

## Run it

You can see various options by issuing the following command

```bash
python main.py --help
```

You will see that there are options to specify various paths (lfw-dir, models-dir and out-dir) as well as
options related to number of classes to process etc

Here is an example command -

```bash
# This command will generate the embeddings for 20 classes where every class in LFW
# has *atleast* 10 images
python main.py dlib --lfw-dir <path_to_lfw>/lfw --models-dir ./models --max-num-classes 20 --min-images-per-class 10 --out-dir <path_to_dir_where_you_want_the_output_file>
```
