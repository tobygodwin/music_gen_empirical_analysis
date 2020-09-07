import tensorflow as tf
import numpy as np
import os
import ddsp
import h5py
import datetime
import tensorflow_io as tfio
import gc
from tensorboard.plugins.hparams import api as hp


class AutoregressiveRNN(tf.keras.Model):

    def __init__(self, rnn_units, out_dim):
        super(AutoregressiveRNN, self).__init__()
        self.out_dim = out_dim
        self.rnn_units = rnn_units

        self.gru = tf.keras.layers.GRU(self.rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform')
        # Dense layers for mu, log_var outputs
        # Linear activation used
        self.fc = tf.keras.layers.Dense(self.out_dim)


        # set up loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse")


    def build_model(self):
        model = tf.keras.Sequential([
            self.gru,
            self.fc
        ])
        return model



def loss(labels,predictions):
    return tf.losses.MeanSquaredError(labels,predictions)


def configure_checkpoints(checkpoint_dir):
    # Configure checkpoints

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)


def get_dataset_hdf5(input_seq_path,target_seq_path,batch_size,feature):
    dataset_size = 100 * 1024
    input_seq = tfio.IODataset.from_hdf5(input_seq_path,dataset='/'+feature+'_input')
    print(input_seq)
    target_seq = tfio.IODataset.from_hdf5(target_seq_path,dataset='/'+feature+'_target')
    print()
    dataset = tf.data.Dataset.zip((input_seq, target_seq))
    print(dataset)
    train_size = int(0.7*dataset_size)
    val_size = int(0.15*dataset_size)
    test_size = int(0.15*dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    return train_dataset, val_dataset, test_dataset




def train(model,train_dataset, val_dataset,log_dir,epochs):

    for input_example_batch, target_example_batch in train_dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=tf.losses.MeanSquaredError())
    checkpoint_callback = configure_checkpoints(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback, tensorboard_callback],
        validation_data=val_dataset
    )
    return history

def train_priors(model_dir):
    # params = ['z', 'f0', 'f0_conf', 'ld']
    params = ['f0', 'f0_conf', 'ld']

    for p in params:
        print(p)
        if p == 'z':
            path_input_seq = f'{model_dir}/z_input.h5'
            path_target_seq = f'{model_dir}/z_target.h5'
        elif p == 'f0':
            path_input_seq = f'{model_dir}/f0_input.h5'
            path_target_seq = f'{model_dir}/f0_target.h5'
        elif p == 'f0_conf':
            path_input_seq = f'{model_dir}/f0_conf_input.h5'
            path_target_seq = f'{model_dir}/f0_conf_target.h5'
        elif p == 'ld':
            path_input_seq = f'{model_dir}/ld_input.h5'
            path_target_seq = f'{model_dir}/ld_target.h5'

        if p == 'z':
            out_dim = 16
        else:
            out_dim = 1
        # parameters

        # Loading up dataset
        train_dataset, val_dataset, test_dataset = get_dataset_hdf5(path_input_seq, path_target_seq,
                                                                    batch_size=batch_size, feature=p)
        model = AutoregressiveRNN(rnn_units, out_dim)
        model = model.build_model()
        dir = f'{model_dir}/summaries/{p}'
        log_dir = f"{dir}/lr:{lr}-batch_size:{batch_size}-epochs:{n_epochs}-rnn_units:{rnn_units}"

        history = train(model, train_dataset, val_dataset, log_dir, epochs=n_epochs)

        results = model.evaluate(test_dataset, batch_size=batch_size)
        with open(f"{log_dir}/results.txt", 'w') as f:
            f.write(f"Total Loss: {results}")

        del model
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == '__main__':
    model = 'model2_20000steps_dset1_reverb'

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256,512,1024]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32,64,128]))
    HP_LR = hp.HParam('lr', hp.Discrete([1e-3,2e-3]))
    HP_EPOCHS = hp.HParam('n_epoch', hp.Discrete([10,30,50]))

    # hyper params
    batch_sizes = [32,64,128]
    n_epochs_list = [10,30,50,60]
    lrs = [2e-3,1e-3]
    rnn_units_list = [512,1024]
    batch_size = 128
    n_epochs = 10
    rnn_units = 1024
    lr = 1e-3

    model_dir = f'/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/activated_pleasant/{model}'
    train_priors(model_dir)







