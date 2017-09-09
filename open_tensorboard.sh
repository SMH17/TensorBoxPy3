#!/bin/bash
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR
echo "TensorBoxPy3"
echo "This script starts Tensorboard, allowing you to visualize the progress."
tensorboard --logdir output
xdg-open localhost:6006