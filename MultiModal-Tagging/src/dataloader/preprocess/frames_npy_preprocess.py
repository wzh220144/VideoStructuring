import os
import tensorflow as tf
import numpy as np

def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncates or pads a tensor to new_size on on a given axis.
    
    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.
  
    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.

    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))
  
    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])
  
    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)
  
    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized

class Preprocess:
    
    def __init__(self, 
                 max_frames,
                 return_frames_num,
                 feat_dim = 128,
                 is_training=False,
                 return_idx = False):
        self.max_frames = max_frames
        self.return_frames_num = return_frames_num
        self.is_training = is_training
        self.return_idx = return_idx
        #self.graph = tf.Graph()
        #with tf.get_default_graph():
        self.feat_dim = feat_dim
        self.frames_placeholder = tf.placeholder(shape=[None,None],dtype=tf.float32)
        self.num_frames = tf.minimum(tf.shape(self.frames_placeholder)[0], self.max_frames)
        self.feature_matrix = resize_axis(self.frames_placeholder,axis=0,new_size=self.max_frames)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

    def __call__(self, frames_npy_fn):
        if os.path.exists(frames_npy_fn):
          frames = np.load(frames_npy_fn)
        else:
          print("!"*100+"\n Warning: file {} not exits".format(frames_npy_fn))
          frames = np.zeros((1, self.feat_dim))
        feature_matrix,num_frames = self.sess.run([self.feature_matrix, self.num_frames],feed_dict={self.frames_placeholder:frames})
        idx = os.path.basename(frames_npy_fn).split('.')[0]
        return_list = []
        return_list.append(feature_matrix)
        if self.return_frames_num:
            return_list.append(num_frames)
        if self.return_idx:
            return_list.append(idx)
        return tuple(return_list)
