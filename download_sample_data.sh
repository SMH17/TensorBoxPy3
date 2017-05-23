#!/bin/bash
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR
echo "TensorBoxPy3"
echo "This scripts downloads the sample data and unzips it."
echo "Downloading..."

mkdir -p data && cd data
wget --continue http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
wget --continue http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
echo "Extracting Inception v2  pretrained data..."
tar xf inception_v2_2016_08_28.tar.gz
echo "Extracting Resnet v2 pretrained data..."
tar xf resnet_v2_101_2017_04_14.tar.gz
mkdir -p overfeat_rezoom

if [[ "$1" == '--travis_tiny_data' ]]; then
    wget --continue http://russellsstewart.com/s/brainwash_tiny.tar.gz
    echo "Extracting brainwash sample files..."
	tar xf brainwash_tiny.tar.gz
    echo "Done."
else
    wget --continue https://stacks.stanford.edu/file/druid:sx925dc9385/brainwash.tar.gz
	wget --continue http://russellsstewart.com/s/tensorbox/overfeat_rezoom/save.ckpt-150000v2
    echo "Extracting brainwash sample files..."
	tar xf brainwash.tar.gz
	echo "Done."
fi

mv brainwash images
