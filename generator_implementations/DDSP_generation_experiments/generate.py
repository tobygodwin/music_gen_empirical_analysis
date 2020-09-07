import tensorflow as tf
from train_prior import AutoregressiveRNN
import ddsp
import numpy as np
import ddsp.training
import gin
import os
from librosa.output import write_wav
LD_RANGE = ddsp.spectral_ops.LD_RANGE
SEQ_LENGTH = 32000

def load_model(model,checkpoint_dir,dim):
    # build_model(model,dim)
    # Evaluation step (generating text using the learned model)
    print(checkpoint_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, SEQ_LENGTH, dim]))
    model.summary()
#
def build_model(model, dim):
    test_preds = model(tf.ones([1, SEQ_LENGTH, dim]))
    model.summary()

def generate_seq(model, input_eval):

    # Number of characters to generate
    num_generate = SEQ_LENGTH

    # Converting our start string to numbers (vectorizing)
    # Empty string to store our results
    seq_generated = input_eval

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        #         print(input_eval.shape)
        predictions = model.predict(input_eval)
        #         print(predictions.shape)

        #         input_eval = tf.concat([input_eval, predictions], axis=1)

        seq_generated = tf.concat([seq_generated, predictions], axis=1)

        input_eval = predictions

    return seq_generated

def scaled_to_herz(conditioning):
    return ddsp.core.midi_to_hz(conditioning['f0_scaled'] * 127.0)

def make_conditioning():
    keys = ['audio','f0_confidence','f0_hz','loudness_db','f0_scaled','ld_scaled','z']
    conditioning = {key:0 for key in keys}
    return conditioning

def scaled_to_db(ld_scaled):
    ld_db = (ld_scaled - 1 ) * LD_RANGE
    return ld_db


if __name__ == '__main__':
    # parameters
    rnn_units = 1024
    # out_dim = 1  # 1 for f0
    # z_dim = 16
    # out_dim = z_dim #z_dim for z

    ld_in = tf.expand_dims(tf.expand_dims(np.load('ld_test.npy')[0:200,:],0),0)
    f0_conf_in = tf.expand_dims(tf.expand_dims(tf.expand_dims(np.load('f0_conf.npy')[0:200],0),0),0)
    print(f0_conf_in.shape)
    z_in = tf.expand_dims(tf.expand_dims(np.load('z_test.npy')[0,0:200,:],0),0)

    f0_in =tf.expand_dims(tf.expand_dims(np.load('f_0_test.npy')[0,0:200,:],0),0)

    model_z = AutoregressiveRNN(rnn_units, out_dim=16).build_model()
    model_f0 = AutoregressiveRNN(rnn_units, out_dim=1).build_model()
    model_f0_conf = AutoregressiveRNN(rnn_units, out_dim=1).build_model()
    model_ld = AutoregressiveRNN(rnn_units, out_dim=1).build_model()
    #
    # load_model(model_z, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/activated_pleasant/model2_20000steps_dset1_reverb/summaries/z/lr:0.001-batch_size:128-epochs:10-rnn_units:1024',dim=16)
    # load_model(model_f0, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/activated_pleasant/model2_20000steps_dset1_reverb/summaries/f0/lr:0.001-batch_size:128-epochs:10-rnn_units:1024',dim=1)
    # load_model(model_f0_conf, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/activated_pleasant/model2_20000steps_dset1_reverb/summaries/f0_conf/lr:0.001-batch_size:128-epochs:10-rnn_units:1024',dim=1)
    # load_model(model_ld, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/activated_pleasant/model2_20000steps_dset1_reverb/summaries/ld/lr:0.001-batch_size:128-epochs:10-rnn_units:1024',dim=1)


    load_model(model_z, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/training_checkpoints/z',dim=16)
    load_model(model_f0, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/training_checkpoints/f0',dim=1)
    load_model(model_f0_conf, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/training_checkpoints/f0_conf',dim=1)
    load_model(model_ld, checkpoint_dir='/home/toby/MSc_AI/IndividualProject/DDSP/prior_learners/training_checkpoints/ld',dim=1)




    input_eval_f0 = tf.random.uniform(
        [1,1,1],
        minval=0,
        maxval=None,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )
    input_eval_z = tf.random.uniform(
        [1, 1, 16],
        minval=0,
        maxval=None,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )

    z_gen = generate_seq(model_z,z_in)[:,:-1,:]
    f0_scaled_gen = generate_seq(model_f0,f0_in)[:,:-1,:]
    f0_conf_gen = generate_seq(model_f0_conf,f0_conf_in)[:,:-1,:]
    ld_scaled_gen = generate_seq(model_ld, ld_in)[:, :-1, :]

    conditioning = make_conditioning()
    conditioning['z'] = z_gen
    conditioning['f0_scaled'] = f0_scaled_gen
    conditioning['f0_hz'] = scaled_to_herz(conditioning)
    conditioning['ld_scaled'] = ld_scaled_gen
    conditioning['loudness_db'] = scaled_to_db(ld_scaled_gen)
    conditioning['f0_confidence'] = tf.squeeze(f0_conf_gen, 2)

    np.save('z_gen.npy',z_gen)
    np.save('f0_scaled_gen.npy', f0_scaled_gen)
    np.save('f0_conf_gen.npy', f0_conf_gen)
    np.save('ld_scaled_gen.npy', ld_scaled_gen)


    # Load model
    SAVE_DIR = '/home/toby/MSc_AI/IndividualProject/DDSP/DDSP_train/models/activated_pleasant/model2_20000steps_dset1_reverb'
    gin_file = os.path.join(SAVE_DIR, 'operative_config-0.gin')
    gin.parse_config_file(gin_file)
    model_gen = ddsp.training.models.Autoencoder()
    model_gen.restore(SAVE_DIR)




    audio_gen = model_gen.decode(conditioning, training=False)
    np.save('audio_gen.npy', audio_gen)
    print(np.array(audio_gen).shape)
    write_wav('test.wav',np.array(audio_gen[0,:]),sr=16000)
    # np.save('audio_gen.npy', audio_gen)




