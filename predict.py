# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

#This file is designed for prediction of bounding boxes of target object. 
#Predictions could be made in two ways: command line or service. 
#For service you can call :func:`initialize` once and call :func:`hot_predict` 
#as many times as it needed to. 
#To use you have to provide image or a folder of images to analize, 
#weights resulting of Tensorbox training, and the related hype file.
#e.g. python3 predict.py data/mypicture.jpg output/overfeat_rezoom_2017_01_06_21.07/save.ckpt-999 hypes/overfeat_rezoom.json                          
#e.g. python3 predict.py data/imagefolderpicturesfolder output/overfeat_rezoom_2017_01_06_21.07/save.ckpt-999 hypes/overfeat_rezoom.json  

import tensorflow as tf
import os, json, subprocess
from optparse import OptionParser

from scipy.misc import imread, imresize
from PIL import Image, ImageDraw

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

print("# TensorBoxPy3: target prediction labeling")

def initialize(weights_path, hypes_path, options):
    """Initialize prediction process.
     
    All long running operations like TensorFlow session start and weights loading are made here.
     
    Args:
        weights_path (string): The path to the model weights file. 
        hypes_path (string): The path to the hyperparameters file. 
        options (dict): The options dictionary with parameters for the initialization process.

    Returns (dict):
        The dict object which contains `sess` - TensorFlow session, `pred_boxes` - predicted boxes Tensor, 
          `pred_confidences` - predicted confidences Tensor, `x_in` - input image Tensor, 
          `hypes` - hyperparametets dictionary.
    """

    H = prepare_options(hypes_path, options)

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas \
            = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
            [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, weights_path)
    return {'sess': sess, 'pred_boxes': pred_boxes, 'pred_confidences': pred_confidences, 'x_in': x_in, 'hypes': H}


def hot_predict(image_path, init_params, options):
    """Makes predictions when all long running preparation operations are made. 
    
    Args:
        image_path (string): The path to the source image. 
        init_params (dict): The parameters produced by :func:`initialize`.
        options (dict): The options for more precise prediction of bounding boxes.

    Returns (Annotation):
        The annotation for the source image.
    """

    H = init_params['hypes']

    # predict
    orig_img = imread(image_path)[:, :, :3]
    img = imresize(orig_img, (H['image_height'], H['image_width']), interp='cubic')
    (np_pred_boxes, np_pred_confidences) = init_params['sess'].\
        run([init_params['pred_boxes'], init_params['pred_confidences']], feed_dict={init_params['x_in']: img})
    pred_anno = al.Annotation()
    pred_anno.imageName = image_path
    _, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=options['show_suppressed'])

    pred_anno.rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2]
    pred_anno.imagePath = os.path.abspath(image_path)
    pred_anno = rescale_boxes((H['image_height'], H['image_width']), pred_anno, orig_img.shape[0], orig_img.shape[1])
    return pred_anno


def prepare_options(hypes_path, options):
    """Sets parameters of the prediction process.
        
    Args:
        hypes_path (string): The path to model hyperparameters file.
        options (dict): The command line options to set before start predictions.

    Returns (dict):
        The model hyperparameters dictionary.
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu'])
    with open(hypes_path, 'r') as f:
        H = json.load(f)
    return H


def save_results(image_path, anno):
    """Saves results of the prediction.
    
    Args:
        image_path (string): The path to source image to predict bounding boxes.
        anno (Annotation): The predicted annotations for source image.

    Returns: 
        Nothing.
    """

    # draw
    new_img = Image.open(image_path)
    d = ImageDraw.Draw(new_img)
    for r in anno.rects:
        d.rectangle([r.left(), r.top(), r.right(), r.bottom()], outline=(0, 255, 0))
    detections_count=len(anno.rects)
    if detections_count>0:
        print("Number of target detections:", detections_count)
    else:
        print("Target hasn't been detected.")

    # save
    output_path=os.path.dirname(image_path)+os.path.sep+'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fpath = os.path.join(output_path, os.path.basename(image_path)+'_result.png')
    new_img.save(fpath)
    subprocess.call(['chmod', '777', fpath])

    fpath = os.path.join(output_path, os.path.basename(image_path)+'_result.json')
    al.saveJSON(fpath, anno)
    subprocess.call(['chmod', '777', fpath])


def main():
    parser = OptionParser(usage='usage: %prog [options] <image> <weights> <hypes>')
    parser.add_option('--gpu', action='store', type='int', default=0)
    parser.add_option('--tau', action='store', type='float',  default=0.25)
    parser.add_option('--min_conf', action='store', type='float', default=0.2)
    parser.add_option('--show_suppressed', action='store_true', dest='show_suppressed', default=False)
    
    (options, args) = parser.parse_args()
    if len(args) < 3:
        print ('You have to provide 3 parameters: image or image directory, weights(save.ckpt) and hypes(hype.json) paths')
        return

    init_params = initialize(args[1], args[2], options.__dict__)
    if os.path.isdir(args[0]):
        print("Detecting target in all the pictures...")
        for filename in os.listdir(args[0]):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                current_image=os.path.join(args[0], filename)
                print("Detecting target in the picture:", filename)
                pred_anno = hot_predict(current_image, init_params, options.__dict__)
                save_results(current_image, pred_anno)
            else:
                print("Skipped file:",os.path.join(args[0], filename))
    else:
        print("Detecting target in the picture...")
        pred_anno = hot_predict(args[0], init_params, options.__dict__)
        save_results(args[0], pred_anno)
    print("Prediction output saved in the same folder of:",args[0])

if __name__ == '__main__':
    main()
