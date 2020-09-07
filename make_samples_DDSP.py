# Code primarily taken from
# https://github.com/magenta/ddsp/blob/master/ddsp/colab/demos/train_autoencoder.ipynb

import ddsp
from ddsp.training import (data)
from librosa.output import write_wav
import random
import ddsp.training
import gin
import numpy as np
import os
import gc

def make_path(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)
    return pth


def setup_paths(SAVE_DIR, LABEL_NAME):
    SAMPLES = make_path(os.path.join(SAVE_DIR,'samples'))
    RECONSTRUCTED = make_path(os.path.join(SAMPLES,'reconstructed'))
    ORIGINAL = make_path(os.path.join(SAMPLES,'original'))
    RECONSTRUCTED_LABEL = make_path(os.path.join(RECONSTRUCTED,LABEL_NAME))
    ORIGINAL_LABEL = make_path(os.path.join(ORIGINAL, LABEL_NAME))
    return RECONSTRUCTED_LABEL, ORIGINAL_LABEL

def setup_dataset(TRAIN_TFRECORD_FILEPATTERN):
    data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
    dataset = data_provider.get_batch(batch_size=10, shuffle=False)

    try:
        batch = next(iter(dataset))
    except OutOfRangeError:
        raise ValueError(
          'TFRecord contains no examples. Please try re-running the pipeline with '
          'different audio file(s).')
    return dataset


def load_model(SAVE_DIR, sample_save_dir):
    if not os.path.exists(sample_save_dir):
        os.makedirs(sample_save_dir)

    gin_file = os.path.join(SAVE_DIR, 'operative_config-0.gin')
    gin.parse_config_file(gin_file)

    # Load model
    model = ddsp.training.models.Autoencoder()
    model.restore(SAVE_DIR)
    return model


def audio_gen(model,reconstructed_path,original_path,ctr,x,batch):
    name = f'{original_path}/{ctr}_extra_original.wav'
    audio = np.array(batch['audio'][x, :])
    write_wav(name, audio, 16000)

    conditioning = model.encode(batch, training=False)
    audio_gen = model.decode(conditioning, training=False)
    audio = np.array(audio_gen[x, :])
    name = f'{reconstructed_path}/{ctr}_extra_reconstructed.wav'
    write_wav(name, audio, 16000)


def make_samples(num_samples,DSET_PATH,LABEL_NAME,model):

    # model = 'model3_20000steps_dset1_1024ch_rnn'

    TRAIN_TFRECORD_FILEPATTERN = f'{DSET_PATH}/train.tfrecord*'
    SAVE_DIR = f'/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/models/{GEN_MODEL_NAME}/{model}'
    sample_save_dir = f'{SAVE_DIR}/samples'
    print(sample_save_dir)
    dataset = setup_dataset(TRAIN_TFRECORD_FILEPATTERN)
    print(SAVE_DIR)
    model = load_model(SAVE_DIR, sample_save_dir)
    dset = iter(dataset)

    reconstructed_path, original_path = setup_paths(SAVE_DIR, LABEL_NAME)
    print(reconstructed_path,original_path)
    ctr = 0
    items = []
    done=False

    for batch in dset:
        if not done:
            for x in range(10):
                if random.choices([True, False], weights=[0.0005, 0.9995])[0]:
                    if ctr == num_samples:
                        done = True
                        break
                    ctr += 1
                    audio_gen(model,reconstructed_path,original_path,ctr,x,batch)
        if done:
            break
    del dset
    del dataset
    gc.collect()

# sexy, fast, groovy, travel, holiday, 
if __name__ == '__main__':
    num_samples = 5600
    GEN_MODEL_NAME = 'all_default_labels'
    # parent_path = '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/dset2_default_labels/tfrecords'
    models = ['model2_20000steps_dset1_reverb']
    # labels = os.listdir(parent_path)
    # labels = ['deactivated_pleasant']
    labels = ['sexy','fast','groovy','travel','holiday']
    # class_dirs = [os.path.join(parent_path,x) for x in labels]
    class_dirs = ['/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/generation_dsets/default_labels/gset1/sexy/tfrecords',
    '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/generation_dsets/default_labels/gset1/fast/tfrecords',
    '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/generation_dsets/default_labels/gset1/groovy/tfrecords',
    '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/generation_dsets/default_labels/gset1/travel/tfrecords',
    '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/generation_dsets/default_labels/gset1/holiday/tfrecords'
    ]
    for idx,DSET_PATH in enumerate(class_dirs):
        LABEL_NAME = labels[idx]
        # GEN_MODEL_NAME = LABEL_NAME
        for model in models:
            make_samples(num_samples, DSET_PATH, LABEL_NAME, model)




    # DSET_PATH = '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/dset2_default_labels/tfrecords'
    #
    # make_samples(num_samples,DSET_PATH,LABEL_NAME,model)
