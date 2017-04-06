echo "TensorBoxPy3"
echo "This script downloads the sample data and unzips it. If you haven't tar for Windows in your user path. Try to unzip files manually."
echo "Downloading..."

mkdir data
cd data\
powershell -Command "Invoke-WebRequest http://russellsstewart.com/s/tensorbox/inception_v1.ckpt  -OutFile inception_v1.ckpt"
powershell -Command "Invoke-WebRequest http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz -OutFile resnet_v1_101_2016_08_28.tar.gz"
mkdir overfeat_rezoom 
echo "Extracting Resnet v1 data..."
tar xf resnet_v1_101_2016_08_28.tar.gz

IF "%1" == "travis_tiny_data"( 
    powershell -Command "Invoke-WebRequest http://russellsstewart.com/s/brainwash_tiny.tar.gz -OutFile brainwash_tiny.tar.gz"
    echo "Extracting brainwash files..."
	tar xf brainwash_tiny.tar.gz
    echo "Done." ) 
ELSE (
    powershell -Command "Invoke-WebRequest  https://stacks.stanford.edu/file/druid:sx925dc9385/brainwash.tar.gz -OutFile brainwash.tar.gz"
	powershell -Command "Invoke-WebRequest  http://russellsstewart.com/s/tensorbox/overfeat_rezoom/save.ckpt-150000v2 -OutFile save.ckpt-150000v2"
	echo "Extracting brainwash files..."
	tar xf brainwash.tar.gz
	echo "Done." )
	
pause
