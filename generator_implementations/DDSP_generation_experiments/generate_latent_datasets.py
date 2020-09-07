import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import time
from IPython.display import Audio
import ddsp
from ddsp.training import (data, decoders, encoders, models, preprocessing,
                           train_util, trainers,eval_util)
import glob
import os
import ddsp.training
import gin
from matplotlib import pyplot as plt
import numpy as np
import h5py

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)

# models = [
# 'model1_20000steps_dset1',
# 'model2_20000steps_dset1_reverb',
# 'model3_20000steps_dset1_1024ch_rnn',
# 'model4_20000steps_abs',
# 'model6_20000steps_reverb_1024rnn',
# 'model7_20000steps_abs_med_reverb',
# 'model8_20000steps_abs_med_reverb_5rnn'
# ]

models = [
'model1_20000steps_dset1',
]



BATCH_SIZE = 256
num_batches = 400  # num_batches = num_batches * batch_size

TRAIN_TFRECORD = '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/dset1/train.tfrecord'
TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'

# Loading Dataset
data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_batch(batch_size=BATCH_SIZE, shuffle=False)
dataset_iter = iter(dataset)

for model in models:
    print(f"Processing mode: {model}")
    dset_save_dir = f'/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/{model}'
    if not os.path.exists(dset_save_dir):
        os.mkdir(dset_save_dir)


    SAVE_DIR = f'/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/{model}'
    # Load model
    gin_file = os.path.join(SAVE_DIR, 'operative_config-0.gin')
    gin.parse_config_file(gin_file)
    model = ddsp.training.models.Autoencoder()
    model.restore(SAVE_DIR)


    def create_h5(name,size):
        name_file = h5py.File(dset_save_dir + '/' + name +'.h5','w')
        name_array = name_file.create_dataset(name,size)
        return name_array

    #Target is sequence [1:]
    z_target = create_h5('z_target',(num_batches*BATCH_SIZE, 1000-1, 16))  # [number_files, seq length, num_channels]
    f0_target = create_h5('f0_target',(num_batches*BATCH_SIZE, 1000-1, 1))
    ld_target = create_h5('ld_target',(num_batches*BATCH_SIZE, 1000-1, 1))
    f0_conf_target = create_h5('f0_conf_target',(num_batches*BATCH_SIZE, 1000-1, 1))

    #inputs are sequence [-1]
    z_input = create_h5('z_input',(num_batches*BATCH_SIZE, 1000-1, 16))
    f0_input = create_h5('f0_input',(num_batches*BATCH_SIZE, 1000-1, 1))
    ld_input = create_h5('ld_input',(num_batches*BATCH_SIZE, 1000-1, 1))
    f0_conf_intput = create_h5('f0_conf_input',(num_batches*BATCH_SIZE, 1000-1, 1))
    print(dataset.element_spec)

    idx = 0
    done = False
    for batch in dataset_iter:
        if done:
            break
        if idx == num_batches:
            done = True

        conditioning = model.encode(batch, training=False)
        i = idx * BATCH_SIZE
        f0_input[i:i + BATCH_SIZE, :, :] = conditioning['f0_scaled'][:, :-1, :]
        z_input[i:i + BATCH_SIZE, :, :] = conditioning['z'][:, :-1, :]
        ld_input[i:i + BATCH_SIZE, :, :] = conditioning['ld_scaled'][:, :-1, :]
        f0_conf_intput[i:i + BATCH_SIZE, :, :] = tf.expand_dims(conditioning['f0_confidence'][:, :-1], 2)

        f0_target[i:i + BATCH_SIZE, :, :] = conditioning['f0_scaled'][:, 1:, :]
        z_target[i:i + BATCH_SIZE, :, :] = conditioning['z'][:, 1:, :]
        ld_target[i:i + BATCH_SIZE, :, :] = conditioning['ld_scaled'][:, 1:, :]
        f0_conf_target[i:i + BATCH_SIZE, :, :] = tf.expand_dims(conditioning['f0_confidence'][:, 1:], 2)
        idx += 1

    print(ld_target.shape)