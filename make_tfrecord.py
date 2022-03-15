import tensorflow as tf
import numpy as np
import glob
import json
from PIL import Image
from tqdm import tqdm

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
img_directory = '/home/co0122/PROJECT/SLABEDGE/DATA/IMAGE/'
label_directory = '/home/co0122/PROJECT/SLABEDGE/DATA/LABEL/'
img_path_list = glob.glob(img_directory+'*.tif')
label_path_list = glob.glob(label_directory+'*.json')
sample_list = [x.split('/')[-1][:-4] for x in img_path_list]

with open(label_path_list[0]) as json_file:
    label = json.load(json_file)
    
label['shapes'][0].keys()

def serialize_example(image, width, height, points, segmentation, segmentation_extra):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'image': _bytes_feature(tf.io.serialize_tensor(image)),
      'width': _int64_feature(width),
      'height': _int64_feature(height),
      'points': tf.train.Feature(int64_list=tf.train.Int64List(value=points)),
      'segmentation': _bytes_feature(tf.io.serialize_tensor(segmentation)),
      'segmentation_extra': _bytes_feature(tf.io.serialize_tensor(segmentation_extra)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
  
sample = sample_list[0]
image_path = img_directory+sample+'.tif'
label_path = label_directory+sample+'.json'
with open(label_path) as json_file:
    label = json.load(json_file)
    
image = np.asarray(Image.open(image_path))
width = image.shape[1]
height = image.shape[0]
points = [label['shapes'][0]['points'][0][0], label['shapes'][0]['points'][1][0]]
segmentation = np.array(label['shapes'][0]['segmentations']).astype(np.int64)
segmentation_extra = np.array(label['shapes'][0]['segmentations_extra']).astype(np.int64)

serialized_example = serialize_example(image, width, height, points, segmentation, segmentation_extra)

# Write the `tf.train.Example` observations to the file.
for sample in tqdm(sample_list):
    image_path = img_directory+sample+'.tif'
    label_path = label_directory+sample+'.json'
    with open(label_path) as json_file:
        label = json.load(json_file)
        
    image = np.asarray(Image.open(image_path))
    width = image.shape[1]
    height = image.shape[0]
    points = [label['shapes'][0]['points'][0][0], label['shapes'][0]['points'][1][0]]
    segmentation = np.array(label['shapes'][0]['segmentations']).astype(np.int64)
    segmentation_extra = np.array(label['shapes'][0]['segmentations_extra']).astype(np.int64)
    
    path = '/home/co0122/PROJECT/SLABEDGE/DATA/tfrecords/'+sample+'.tfrecord'
    
    with tf.io.TFRecordWriter(path) as writer:
        serialized_example = serialize_example(image, width, height, points, segmentation, segmentation_extra)
        writer.write(serialized_example)
        
tfrecord_directory = '/home/co0122/PROJECT/SLABEDGE/DATA/tfrecords/'

tfrecord_path_list = glob.glob(tfrecord_directory+'*')
train_path_list = tfrecord_path_list[:int(len(tfrecord_path_list)*0.85)]
test_path_list = tfrecord_path_list[int(len(tfrecord_path_list)*0.85):]

ds_train = tf.data.TFRecordDataset(train_path_list)
ds_test = tf.data.TFRecordDataset(test_path_list)

# Create a description of the features.
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'points': tf.io.FixedLenFeature((2,), tf.int64,),
    'segmentation': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'segmentation_extra': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def _decode_image(example):
    example['image'] = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    example['image'] = tf.cast(example['image'], dtype=tf.float32)
    example['image'].set_shape([176,1600])
    example['segmentation'] = tf.io.parse_tensor(example['segmentation'], out_type=tf.int64)
    example['segmentation'] = tf.cast(example['segmentation'], dtype=tf.int32)
    example['segmentation'].set_shape([176,1600])
    return example

def _expand_channel_dim(example):
    example['image'] = tf.expand_dims(example['image'], axis=-1)
    return example
# def _onehot_segmentation(example):
#     example['segmentation'] = tf.one_hot(example['segmentation'], depth=2)
#     return example

def _return_xy(example):
    return example['image'], example['segmentation']
  
ds_train = ds_train.map(_parse_function)
ds_train = ds_train.map(_decode_image)
ds_train = ds_train.map(_expand_channel_dim)
# ds_train = ds_train.map(_onehot_segmentation)
ds_train = ds_train.map(_return_xy)

ds_test = ds_test.map(_parse_function)
ds_test = ds_test.map(_decode_image)
ds_test = ds_test.map(_expand_channel_dim)
# ds_train = ds_train.map(_onehot_segmentation)
ds_test = ds_test.map(_return_xy)

ds_train = ds_train.shuffle(buffer_size=500)
ds_train = ds_train.batch(1)
