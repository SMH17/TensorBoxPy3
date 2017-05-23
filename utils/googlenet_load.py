# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

from utils.slim_nets import inception
from utils.slim_nets import resnet
import tensorflow.contrib.slim as slim


def model(x, H, reuse, is_training=True):
    slim_attention_lname='Mixed_3b'

    if H['slim_basename'] == 'resnet_v1_101':
        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, T = resnet.resnet_v1_101(x,
                                    is_training=is_training,
                                    num_classes=1000,
                                    reuse=reuse)
    elif H['slim_basename'] == 'resnet_v2_152':
        with slim.arg_scope(resnet.resnet_arg_scope()):
            _, T = resnet.resnet_v2_101(x,
                                    is_training=is_training,
                                    num_classes=1001,
                                    reuse=reuse)
    elif H['slim_basename'] == 'InceptionV1':
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            _, T = inception.inception_v1(x,
                                          is_training=is_training,
                                          num_classes=1001,
                                          spatial_squeeze=False,
                                          reuse=reuse)
    elif H['slim_basename'] == 'InceptionV2':
        with slim.arg_scope(inception.inception_v2_arg_scope()):
            _, T = inception.inception_v2(x,
                                          is_training=is_training,
                                          num_classes=1001,
                                          spatial_squeeze=False,
                                          reuse=reuse)
    elif H['slim_basename'] == 'InceptionV3':
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, T = inception.inception_v3(x,
                                          is_training=is_training,
                                          num_classes=1001,
                                          spatial_squeeze=False,
                                          reuse=reuse)
        slim_attention_lname='Mixed_5b'
    elif H['slim_basename'] == 'InceptionV4':
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            _, T = inception.inception_v4(x,
                                          is_training=is_training,
                                          num_classes=1001,
                                          reuse=reuse)
        slim_attention_lname='Mixed_4a'
            # print '\n'.join(map(str, [(k, v.op.outputs[0].get_shape()) for k, v in T.iteritems()]))

    coarse_feat = T[H['slim_top_lname']][:, :, :, :H['later_feat_channels']]
    assert coarse_feat.op.outputs[0].get_shape()[3] == H['later_feat_channels']

    # fine feat can be used to reinspect input
    attention_lname = H.get('slim_attention_lname', slim_attention_lname)
    early_feat = T[attention_lname]

    return coarse_feat, early_feat
