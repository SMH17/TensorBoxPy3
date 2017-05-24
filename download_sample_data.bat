echo "TensorBoxPy3"
echo "This script downloads and unzips the pretrained submodel data and labeled images of brainwash sample used in the original TensorBox presentation. If you haven't tar for Windows in your user path. Try to unzip files manually."
echo "Downloading..."

mkdir data
cd data\
powershell -Command "Invoke-WebRequest http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz  -OutFile inception_v2.tar.gz"
powershell -Command "Invoke-WebRequest http://download.tensorflow.org/models/resnet_v2_152_2017_04_14tar.gz -OutFile resnet_v2.tar.gz"
mkdir overfeat_rezoom
echo "Extracting Inception v2 pretrained data..."
tar xf inception_v2_2016_08_28.tar.gz
echo "Extracting Resnet v2 pretrained data..."
tar xf resnet_v2_152_2017_04_14.tar.gz

IF "%1" == "travis_tiny_data"(
    powershell -Command "Invoke-WebRequest http://russellsstewart.com/s/brainwash_tiny.tar.gz -OutFile brainwash_tiny.tar.gz"
    echo "Extracting brainwash sample files..."
    tar xf brainwash_tiny.tar.gz
    echo "Done." )
ELSE (
    powershell -Command "Invoke-WebRequest  https://stacks.stanford.edu/file/druid:sx925dc9385/brainwash.tar.gz -OutFile brainwash.tar.gz"
	  echo "Extracting brainwash sample files..."
	  tar xf brainwash.tar.gz
	  echo "Done." )

rename brainwash images

pause
