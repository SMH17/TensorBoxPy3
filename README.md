## TensorBoxPy 3

TensorBox fork by Silvio Marano.

Compared to the original TensorBox release by russel91 for Python 2.6 and Tensorflow 0.11
this fork is aimed to support only the newer Python 3.x and Tensorflow 1.x versions, in addition
includes some minor fix to avoid crash of evaluation function and improve the Windows support,
and merges features of others branches not committed in original branch.
Additional features and utility will be added soon and development will diverge from original release
update after update, so pay attention if you try to mix code from TensorBoxPy3 with code of the
original TensorBox branch.
Please check requirements file before try to run.

Before run, compile stitch_wrapper with:

    $ cd utils && make && cd ..

If you want to use lstm hype for ReInspect Neural Network:
add cuDNN to your path, in Linux put the libcudnn*.so files on your LD_LIBRARY_PATH e.g.

    $ cp /path/to/appropriate/cudnn/lib64/* /usr/local/cuda/lib64

compile also libhungarian with:

    $ cd utils && make && make hungarian && cd ..

ReInspect is a neural network extension to Overfeat-GoogLeNet designed for high performance object detection in images with heavily overlapping instances.

## Usage example:

To extract frames from a video use command:

    $ python3 extract_video_frames.py yourvideofile.mp4

To train use command:

    $ python3 train.py --hypes hypes/overfeat_rezoom.json --max_iter=20000 --logdir output

To evaluate use command:

    $ python3 evaluate.py --weights output/overfeat_rezoom_2017_05_17.10/save.ckpt-19999 --test_boxes data/weapons/validation.json

To predict use command:

    $ python3 predict.py data/images/ output/overfeat_rezoom_2017_05_17.10/save.ckpt-19999  hypes/overfeat_rezoom.json   

To combine frames in a video use command:

    $ python3 combine_frames.py -ext jpg -o output.mp4


In this example is used Overfeat Rezoom with 20000 iterations and the date of training completion is 17/05/2017 17.10, the names of folders and save.ckpt are generated consequently.

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.

    $ cd /path/to/tensorbox
    $ tensorboard --logdir output
    $ # (optional, start an ssh tunnel if not experimenting locally)
    $ ssh myserver -N -L localhost:6006:localhost:6006
    $ # open localhost:6006 in your browser

The original project [link](https://github.com/Russell91/TensorBox/).
