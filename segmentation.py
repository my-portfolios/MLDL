import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import cv2
import json
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

img_directory = '/home/co0122/PROJECT/SLABEDGE/DATA/IMAGE/'
label_directory = '/home/co0122/PROJECT/SLABEDGE/DATA/LABEL/'
sample_list = [x.split('/')[-1][:-4] for x in glob.glob(img_directory+'*.tif')]

TEST_RATIO = 0.15

train_samples = sample_list[:-int(len(sample_list)*TEST_RATIO)]
test_samples = sample_list[-int(len(sample_list)*TEST_RATIO):]

train_img_list = []
train_label_list = []
test_img_list = []
test_label_list = []

for sample in tqdm(train_samples):
    img_path = img_directory+sample+'.tif'
    label_path = label_directory+sample+'.json'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    with open(label_path) as json_file:
        label = json.load(json_file)['shapes'][0]
    train_img_list.append(np.expand_dims(img, axis=-1))
    train_label_list.append(np.array(label['segmentations']))
#     train_label_list.append(np.array(label['segmentations_extra']))

for sample in tqdm(test_samples):
    img_path = img_directory+sample+'.tif'
    label_path = label_directory+sample+'.json'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    with open(label_path) as json_file:
        label = json.load(json_file)['shapes'][0]
    test_img_list.append(np.expand_dims(img, axis=-1))
    test_label_list.append(np.array(label['segmentations']))
#     test_label_list.append(np.array(label['segmentations_extra']))

ds_train = tf.data.Dataset.from_tensor_slices((train_img_list, train_label_list))
ds_test = tf.data.Dataset.from_tensor_slices((test_img_list, test_label_list))

rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255)
])

def one_hot(y):
    return tf.one_hot(tf.cast(y, tf.int32), 3)

BATCH_SIZE = 1
IMG_SIZE = (176, 1600)
seed=25
AUTOTUNE = tf.data.experimental.AUTOTUNE
img_flip = tf.keras.layers.RandomFlip(seed=seed, mode= "vertical")
mask_flip = tf.keras.layers.RandomFlip(seed=seed, mode= "vertical")
img_trans = tf.keras.layers.RandomTranslation(0, (-0.05, 0.15), fill_mode='nearest',interpolation='nearest', seed=seed)
mask_trans = tf.keras.layers.RandomTranslation(0, (-0.05, 0.15), fill_mode='nearest',interpolation='nearest', seed=seed)

def preprocessing(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (rescale(x), y))
#     ds = ds.map(lambda x, y: (x, one_hot(y)))
    
    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(BATCH_SIZE)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (img_flip(x, training=True), mask_flip(y[...,tf.newaxis], training=True)), 
                    num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (img_trans(x, training=True), mask_trans(y, training=True)), 
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)

ds_train = preprocessing(ds_train, shuffle=True, augment=True)
ds_test = preprocessing(ds_test)

from IPython.display import clear_output

def display(display_list):
    plt.figure(figsize=(10, 4.5))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(len(display_list), 1, i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            mask = mask[...,tf.newaxis]
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
#         clear_output(wait=True)
        show_predictions(ds_test)
        print ('\n에포크 이후 예측 예시 {}\n'.format(epoch+1))
        
# Encoder
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    '''
    Add 2 convolutional layers with the parameters
    '''
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x

def encoder_block(inputs, n_filters=16, pool_size=(2,2), dropout=0.0):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    f = conv2d_block(inputs, n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
#     p = tf.keras.layers.Dropout(dropout)(p)
    return f, p

def encoder(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    f1, p1 = encoder_block(inputs, n_filters=32, pool_size=(2,2))
    f2, p2 = encoder_block(p1, n_filters=64,pool_size=(2,2))
    f3, p3 = encoder_block(p2, n_filters=128)
#     f4, p4 = encoder_block(p3, n_filters=256)
#     return p4, (f1, f2, f3, f4)
    return p3, (f1, f2, f3)

# Bottlenect
def bottleneck(inputs):
#     bottle_neck = conv2d_block(inputs, n_filters=512)
    bottle_neck = conv2d_block(inputs, n_filters=256)
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.0):
    '''
    defines the one decoder block of the UNet
    '''
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
#     c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters)
    return c
def decoder(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.
    '''
#     f1, f2, f3, f4 = convs
#     c6 = decoder_block(inputs, f4, n_filters=256, kernel_size=3, strides=2)
#     c7 = decoder_block(c6, f3, n_filters=128, kernel_size=3, strides=2)
#     c8 = decoder_block(c7, f2, n_filters=64, kernel_size=3, strides=(2, 2))
#     c9 = decoder_block(c8, f1, n_filters=32, kernel_size=3, strides=(2, 2))
#     outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(c9)
    f1, f2, f3 = convs
    c6 = decoder_block(inputs, f3, n_filters=128, kernel_size=3, strides=2)
    c7 = decoder_block(c6, f2, n_filters=64, kernel_size=3, strides=2)
    c8 = decoder_block(c7, f1, n_filters=32, kernel_size=3, strides=(2, 2))
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(c8)
    return outputs

# Encoder
def conv2d_block(input_tensor, n_filters, kernel_size=9):
    '''
    Add 2 convolutional layers with the parameters
    '''
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x

def encoder_block(inputs, n_filters=16, pool_size=(2,2), dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    f = conv2d_block(inputs, n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)
    return f, p

def encoder(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    f1, p1 = encoder_block(inputs, n_filters=32, pool_size=(2,4))
    f2, p2 = encoder_block(p1, n_filters=64,pool_size=(2,4))
    f3, p3 = encoder_block(p2, n_filters=128)
    f4, p4 = encoder_block(p3, n_filters=256)
    return p4, (f1, f2, f3, f4)

# Bottlenect
def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=512)
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    '''
    defines the one decoder block of the UNet
    '''
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters)
    return c
def decoder(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.
    '''
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=256, kernel_size=9, strides=2)
    c7 = decoder_block(c6, f3, n_filters=128, kernel_size=9, strides=2)
    c8 = decoder_block(c7, f2, n_filters=64, kernel_size=9, strides=(2, 4))
    c9 = decoder_block(c8, f1, n_filters=32, kernel_size=9, strides=(2, 4))
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(c9)
    return outputs

# putting it all together
OUTPUT_CHANNELS = 2
def UNet():
    '''
    Defines the UNet by connecting the encoder, bottleneck and decoder
    '''
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[0],IMG_SIZE[1],1,))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs, outputs)
    return model

model = UNet()
model.summary()

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc'])

checkpoint_path = "./checkpoint/segmentation0302_original_seg.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)

# # Loads the weights
# checkpoint_path = "../notebooks/checkpoint/segmentation0223_original_seg.ckpt"
# model.load_weights(checkpoint_path)# Re-evaluate the model

# configure the training parameters and train the model
EPOCHS = 20

model_history = model.fit(ds_train, epochs=EPOCHS,
                          validation_data=ds_test,
                          callbacks=[cp_callback, DisplayCallback()])

checkpoint_path = "./checkpoint/segmentation0304.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc'])

# configure the training parameters and train the model
EPOCHS = 50

model_history = model.fit(ds_train, epochs=EPOCHS,
                          validation_data=ds_test,
                          callbacks=[cp_callback, DisplayCallback()])

# Loads the weights
checkpoint_path = "./checkpoint/segmentation0304.ckpt"
model.load_weights(checkpoint_path)# Re-evaluate the model

show_predictions(ds_test, num=69)

import time
start = time.time()
for x, y in ds_test:
    tmp = model.predict(x)
end = time.time()
print(end-start)

(end-start)/69

def save_image(display_list, count, directory):
    plt.figure(figsize=(10, 4.5))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(len(display_list), 1, i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig(directory+f'/test_{count}.jpg')
    
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def save_predictions(directory, dataset=None, num=1):
    if dataset:
        count=0
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            mask = mask[...,tf.newaxis]
            save_image([image[0], mask[0], create_mask(pred_mask)], count, directory)
            count+=1
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
       
save_predictions('./results/0304', ds_test, num=69)

for x, y in ds_test:
    break
    
from PIL import Image

pred

count=0
for x, y in ds_test:
    pred = model(x)
    pred = pred.numpy().argmax(-1)[0]
    pred_ = 1-pred
    pred_ = (pred_*255).astype(np.uint8)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_, 4)
    blank_label = labels[0,0]
    post_processed = (labels!=blank_label).astype(int)
    print(count)
    plt.title('original image')
    plt.imshow(x.numpy()[0])
    plt.show()
    plt.title('prediction: after post process')
    plt.imshow(post_processed)
    plt.show()
    plt.title('prediction: before post process')
    plt.imshow(pred)
    plt.show()
    plt.title('True label')
    plt.imshow(y.numpy()[0])
    plt.show()
    count=count+1
    
count=0
for x, y in ds_test:
    pred = model(x)
    pred = pred.numpy().argmax(-1)[0]
    pred_ = 1-pred
    pred_ = (pred_*255).astype(np.uint8)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_, 4)
    blank_label = labels[0,0]
    post_processed = (labels!=blank_label).astype(int)
    tmp = (post_processed-np.roll(post_processed, 1))
    tmp[:,0]=0
    edge = int((np.arange(1600)*tmp).sum()/176)
    edge_arr = np.zeros((176,1600))
    edge_arr[:,edge-1:edge+1]=1
    print(count)
    plt.figure(figsize=(16,16))
    plt.title('Original image + vertical line (with additional post processing)')
    plt.imshow(x.numpy()[0])
    plt.imshow(edge_arr, vmin = 0, vmax = 1, interpolation = 'nearest', cmap='Greys', alpha=0.5)
    plt.show()
    plt.figure(figsize=(16,16))
    plt.title('True label + vertical line (with additional post processing)')
    plt.imshow(y.numpy()[0])
    plt.imshow(edge_arr, vmin = 0, vmax = 1, interpolation = 'nearest', cmap='Greys', alpha=0.5)
    plt.show()
    count=count+1
