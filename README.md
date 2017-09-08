## TensorBoxPy 3

TensorBox fork by Silvio Marano.

Compared to the original TensorBox release for Python 2.6 and older Tensorflow versions, this fork is aimed to support the newer Python 3.x and latest Tensorflow 1.x versions, in addition It has: additional fixes to avoid crash of evaluation function, Windows support, merges features of others branches not committed in original branch, uses newer neural networks to improve detection, and includes a prediction functions and additional tools to process videos and improves usability.
More features and utilities will be added in future and the development will continue to diverge from original release update after update, so pay attention if you try to mix code from TensorBoxPy 3 with code of the original branch.
Please check requirements file before trying to run.

Before run, compile stitch_wrapper with:

    $ cd utils && make && cd ..

If you want to use spatial 2D-LSTM(Long short-term memory) for Recurrent Neural Networks you need additional steps:

compile also libhungarian with:

    $ cd utils && make && make hungarian && cd ..

The current hyperparameters configurations work with Inception v2 and ReInspect(ResNet) v2 as sub-models.
- Inception is the evolution of GoogLeNet neural network. Inception relies on a network-in-network architecture that uses sub-networks called Inception modules. The goal of the inception module is to act as a multi-level feature extractor to increase learning abilities and abstraction power by having more complex filters that work at different levels.
- ReInspect uses deep residual networks explicitly reformulating the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions, and relies on micro-architecture modules, using a collection of micro-architecture building blocks along with standard layers (like CONV, POOL, etc.) that together lead to the macro-architecture. ReInspect is a neural network extension designed for high performance object detection in images with heavily overlapping instances.

Both v2 versions of Inception and Resnet sub-models use batch normalization to address covariate shift problem and improve results.


## Usage example:

First of all, you need to define the set of target images where the training tasks have to work. The training needs images labeled with json files that specify what are the target objects bounding boxes. To label images you can pass the images directory to json labeler in this way:

    $ python3 make_json.py images

To extract frames from a video use command:

    $ python3 extract_video_frames.py yourvideofile.mp4

To train use command:

    $ python3 train.py --hypes hypes/inception_rezoom.json --max_iter=20000 --logdir output

To evaluate use command:

    $ python3 evaluate.py --weights output/inception_rezoom_2017_05_17.10/save.ckpt-19999 --test_boxes data/images/validation.json

To predict use command:

    $ python3 predict.py data/images/ output/inception_rezoom_2017_05_17.10/save.ckpt-19999  hypes/inception_rezoom.json   

To combine frames in a video use command:

    $ python3 combine_frames.py -ext jpg -o output.mp4


In this example is used Inception Rezoom with 20000 iterations and the date of training completion is 17/05/2017 17.10, the names of folders and save.ckpt are generated consequently.

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.

    $ cd /path/to/tensorbox
    $ tensorboard --logdir output
    $ # (optional, start an ssh tunnel if not experimenting locally)
    $ ssh myserver -N -L localhost:6006:localhost:6006
    $ # open localhost:6006 in your browser

TensorBoxPy 3 official GitHub repository [here](https://github.com/SMH17/TensorBoxPy3).
The original project [link](https://github.com/Russell91/TensorBox/).
