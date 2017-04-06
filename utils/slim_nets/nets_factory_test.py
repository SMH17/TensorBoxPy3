# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

"""Tests for slim.inception."""

import tensorflow as tf

from utils.slim_nets import nets_factory


class NetworksTest(tf.test.TestCase):

  def testGetNetworkFn(self):
    batch_size = 5
    num_classes = 1000
    for net in nets_factory.networks_map:
      with self.test_session():
        net_fn = nets_factory.get_network_fn(net, num_classes)
        # Most networks use 224 as their default_image_size
        image_size = getattr(net_fn, 'default_image_size', 224)
        inputs = tf.random_uniform((batch_size, image_size, image_size, 3))
        logits, end_points = net_fn(inputs)
        self.assertTrue(isinstance(logits, tf.Tensor))
        self.assertTrue(isinstance(end_points, dict))
        self.assertEqual(logits.get_shape().as_list()[0], batch_size)
        self.assertEqual(logits.get_shape().as_list()[-1], num_classes)

if __name__ == '__main__':
  tf.test.main()
