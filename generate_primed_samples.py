import jukebox
import torch as t
import numpy as np
import librosa
import os
import shutil
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model

from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, sample_partial_window, upsample, primed_sample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
rank, local_rank, device = setup_dist_from_mpi()

def initialise_model(model):

    vqvae, *priors = MODELS[model]
    hps_vqvae = setup_hparams(vqvae, dict(sample_length = 1048576))
    # hps_vqvae.restore_vqvae = '/vol/bitbucket/tg919/music_gen_evaluation_framework/sample_RNN_implementations/jukebox/models'
    vqvae = make_vqvae(hps_vqvae, device)
    
    hps_prior = setup_hparams(priors[-1], dict())
    # hps_prior.restore_vqvae = '/vol/bitbucket/tg919/music_gen_evaluation_framework/sample_RNN_implementations/jukebox/models'
    top_prior = make_prior(hps_prior, vqvae, device)

    # top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

    return vqvae, top_prior

def generate_sample(audio_files, sample_length=25):
    sample_length_in_seconds = sample_length # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                           # range work well, with generation time proportional to sample length.
                                           # This total length affects how quickly the model
                                           # progresses through lyrics (model also generates differently
                                           # depending on if it thinks it's in the beginning, middle, or end of sample)

    hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
    print(hps.sample_length)
    # assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

    total_sample_length_in_seconds = 180
    total_lengths = total_sample_length_in_seconds * hps.sr

    metas = [dict(artist = "Darude",
                genre = "Sandstorm",
                total_length = total_lengths,
                offset = 0,
                lyrics = """Hello
                """,
                ),
              ] * hps.n_samples
    labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

    sampling_temperature = .98

    lower_batch_size = 16
    max_batch_size = 1 if model == "5b" else 16
    lower_level_chunk_size = 32
    chunk_size = 16 if model == "5b" else 32
    sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                            chunk_size=lower_level_chunk_size),
                        dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                             chunk_size=lower_level_chunk_size),
                        dict(temp=sampling_temperature, fp16=True,
                             max_batch_size=max_batch_size, chunk_size=chunk_size)]

    priors = [None, None, top_prior]

    top_raw_to_tokens = priors[-1].raw_to_tokens
    duration = (int(hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
    x = load_prompts(audio_files, duration, hps)

    sample_levels = list(range(len(priors)))
    zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
    zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)


def load_prompts(audio_files, duration, hps):
    xs = []
    for audio_file in audio_files:
        x = load_audio(audio_file, sr=hps.sr, duration=duration, offset=0.0, mono=True)
        x = x.T # CT -> TC
        xs.append(x)
    while len(xs) < hps.n_samples:
        xs.extend(xs)
    xs = xs[:hps.n_samples]
    x = t.stack([t.from_numpy(x) for x in xs])
    x = x.to('cuda', non_blocking=True)
    return x


def get_audio_files(path_to_dataset, num_per_class):
    '''
    Assumes each datasample in class folder structure

    args:
    path_to_dataset: path to top level of dataset folder

    returns:
    audio_files: list of list of audio files per class.
    '''
    class_dirs = [x[0] for x in os.walk(path_to_dataset)][1:]
    audio_files = dict()
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        list_ = os.listdir(class_dir)
        list_ = [os.path.join(class_dir, x) for x in list_]
        audio_files_class = np.random.choice(list_, num_per_class, replace=False)
        audio_files[class_name] = audio_files_class

    return audio_files

def chop_audio(audio, s_to_chunk):
    '''
    removes first s_to_chunk elements from generated audio
    '''
    aud, sr = librosa.load(audio) #audio not in s
    num_to_chunk = s_to_chunk / sr
    return aud[0][num_to_chunk:]

def save_wav(audio, save_path):
    librosa.output.write_wav(save_path, audio, sr)


if __name__ == '__main__':
    exp_name = 'default_labels_jukebox'

    # Hyper params
    num_per_class = 58 #Default labels
    # num_per_class = 330 #Emotional Valence
    sample_length = 24
    generated_samples_path = '/vol/bitbucket/tg919/music_gen_evaluation_framework/data/generated_samples/jukebox_default_labels'
    
    # path_to_dataset = '/vol/bitbucket/tg919/Conditional-SampleRNN-master/datasets/mtg-jamendo-emotional-valence-half-subset'
    path_to_dataset = '/vol/bitbucket/tg919/Conditional-SampleRNN-master/datasets/mood-theme-default'

    # set-up hyper parameters
    model = '5b'
    hps = Hyperparams()
    hps.n_samples = 1 if model=='5b' else 8
    hps.name = exp_name
    chunk_size = 16 if model=="5b" else 32
    max_batch_size = 1 if model=="5b" else 16
    hps.levels = 3
    hps.hop_fraction = [.5,.5,.125]
    hps.mode = 'primed'
    hps.prompt_length_in_seconds = 8
    hps.sr = 24000
    # hps.restore_vqvae = '/vol/bitbucket/tg919/music_gen_evaluation_framework/sample_RNN_implementations/jukebox/models'

    vqvae, top_prior = initialise_model(model)

    audio_files = get_audio_files(path_to_dataset, num_per_class)

    for (label,_) in audio_files.items():
        ctr = 0

        label_path = os.path.join(generated_samples_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        # completed = [str(x) for x in range(37)]
        completed = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','45','47','49','50','51','52','53','54','55']
        if label in completed:
            continue
        
        files = audio_files[label]
        for idx, file in enumerate(files):
            audio = [file]
            print(audio)
            generate_sample(audio, sample_length=sample_length)
            print(label)
            load_path0 = hps.name + '/level_2/item_0.wav'
            shutil.move(load_path0,f"{label_path}/jukebox_generated_24khz_{label}_{ctr}.wav" )
            print(f"{label_path}/jukebox_generated_NEW_{label}_{ctr}.wav")
            
            # aud, sr = librosa.load(load_path0) #audio not in s
            # save_wav(aud, f"{label_path}/jukebox_generated_{label}_{ctr}.wav")

            ctr+= 1
            
            # aud0 = chop_audio(load_path0, hps.prompt_length_in_seconds)
            # save_wav(aud0, f"{label_path}/jukebox_generated_{ctr}.wav")
            # load_path1 = hps.name + '/level_2/item_1.wav'
            # ctr += 1
            # aud1 = chop_audio(load_path1, hps.prompt_length_in_seconds)
            # save_wav(aud1, f"{label_path}/jukebox_generated_{ctr}.wav")


    

    # audio_files = ['/vol/bitbucket/tg919/Conditional-SampleRNN-master/datasets/mtg-jamendo-emotional-valence-subset/2/130.wav']

    # generate_sample(audio_files)
