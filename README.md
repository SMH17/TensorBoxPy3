## TensorBoxPy3

TensorboxPy3 is based on ReInspect a neural network extension designed for high performance object detection in images with heavily overlapping instances. Compared to the original TensorBox release for Python 2.6 and older Tensorflow versions, TensorBoxPy3 is aimed to support the newer Python 3.x and latest Tensorflow 1.x versions, in addition It has: additional fixes to avoid crashes of evaluation function, Windows support, merges features of others branches not committed in original branch, uses newer neural networks to improve detection, and includes a prediction functions and additional tools to process videos and improves usability.
More improvements, features and utilities will be added in future and the development will continue to diverge from original release update after update, so pay attention if you try to mix code from TensorBoxPy3 with code of the original branch.
Please check requirements file before trying to run.

Before run, on Linux compile stitch_wrapper with:

    > cd utils && make && cd ..

If you want to use spatial 2D-LSTM(Long short-term memory) for Recurrent Neural Networks you need additional steps:

compile also libhungarian with:

    > cd utils && make && make hungarian && cd ..

on Windows use Cmake command:

    > cmake -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") CMakeLists.txt

to generate a Visual C++ project, then compile It with Visual Studio.

The current hyperparameters configurations work with Inception v2 and ResNet v2 as sub-models.
- Inception is the evolution of GoogLeNet neural network. Inception relies on a network-in-network architecture that uses sub-networks called Inception modules. The goal of the inception module is to act as a multi-level feature extractor to increase learning abilities and abstraction power by having more complex filters that work at different levels.
- Resnet uses deep residual networks explicitly reformulating the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions, and relies on micro-architecture modules, using a collection of micro-architecture building blocks along with standard layers (like CONV, POOL, etc.) that together lead to the macro-architecture.

Both v2 versions of Inception and Resnet sub-models use batch normalization to address covariate shift problem and improve results.


## Usage example:

First of all, you need to define the set of target images where the training tasks have to work. The training needs images labeled with json files that specify what are the target objects bounding boxes. To label images you can pass the images directory to json_labeler in this way:

    > python3 json_labeler.py images

To extract frames from a video use command:

    > python3 extract_video_frames.py yourvideofile.mp4

To train use command:

    > python3 train.py --hypes hypes/inception_rezoom.json --max_iter=20000 --logdir output

To evaluate use command:

    > python3 evaluate.py --weights output/inception_rezoom_2017_05_17.10/save.ckpt-19999 --test_boxes data/images/validation.json

To predict use command:

    > python3 predict.py data/images/ output/inception_rezoom_2017_05_17.10/save.ckpt-19999  hypes/inception_rezoom.json   

To combine frames in a video use command:

    > python3 combine_frames.py -ext png -o output.mp4


## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard. You can do this manually with the following code, or simply launching the related script.

    > cd /path/to/TensorBoxPy3
    > tensorboard --logdir output
      # (optional, start an ssh tunnel if not experimenting locally)
    > ssh myserver -N -L localhost:6006:localhost:6006
      # open localhost:6006 in your browser

![Tensorboard Screenshot 1](https://s17.postimg.cc/o71j00x33/graph.png | width=1300)
![Tensorboard Screenshot 2](https://s17.postimg.cc/wcjky65wf/hystogram.png | width=1300)
![Tensorboard Screenshot 3](https://s17.postimg.cc/add6aywrz/charts.png | width=1300)

TensorBoxPy3 by Silvio Marano, official GitHub repository [here](https://github.com/SMH17/TensorBoxPy3).
The original project [cf-link](https://github.com/Russell91/ReInspect) or [tf-link](https://github.com/Russell91/TensorBox/).
