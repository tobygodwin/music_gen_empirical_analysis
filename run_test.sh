
# Arguments
exp_name="ddsp_model2_default_targeted"


NUM_SAMPLES=196 #number samples per class
FRAME_SIZES="16 4"
NUM_RNN=2


#Making dirs
mkdir /home/toby/MSc_AI/IndividualProject/music_gen_framework/data/generated_samples
mkdir /home/toby/MSc_AI/IndividualProject/music_gen_framework/data/generated_samples/${exp_name}
mkdir /home/toby/MSc_AI/IndividualProject/music_gen_framework/data/generated_samples/${exp_name}/audio
mkdir /home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/melspecs
mkdir /home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier

LOG=data/generated_samples/${exp_name}/results.txt

#PATHS
BASE="/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework"
path_to_generator='/home/toby/MSc_AI/IndividualProject/Conditional-SampleRNN-master/results/exp:default_subset-frame_sizes:16,4-n_rnn:2-dataset:mood-theme-default-nb_classes:56/checkpoints/best-ep3-it206625'

######################## EMOTIONAL VALENCE LABELS #######################

# DATASET=emotional_valence
# path_to_pre_trained_classifier=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/output/pretrained_baseline_emotional_valence.pth

# path_to_results_folder=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/audio

# path_to_dataset=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/
# TRAIN_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs/
# TEST_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs/
# LABELS_TXT=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/emotional_valence.txt
# TRAIN_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_train_labels.csv
# TEST_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_test_labels.csv
# VAL_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_validation_labels.csv
# LOG_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier

########################## DEFAULT LABELS ##############################
DATASET=default_labels
path_to_pre_trained_classifier=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/retrain_baseline_default_NEW_AFTERBUG/classifier/retrain_baseline_default_NEW_AFTERBUG.pth
path_to_results_folder=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/audio
path_to_dataset=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/
TRAIN_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs/
TEST_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs/
LABELS_TXT=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/tags/moodtheme_split.txt
TRAIN_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv
TEST_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv
VAL_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv
LOG_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier

################# REPORTING PERFORMANCE ON TEST SPLIT ##########################
echo "* * * * Reporting test statistics * * * * " >> ${LOG}
echo "Training Data: train split" >> ${LOG}
#Running classifier test
python3 -W ignore \
${BASE}/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/main.py \
test \
${exp_name} \
--train_datapath ${TRAIN_PATH} \
--test_datapath ${TEST_PATH} \
--val_datapath ${VAL_PATH} \
--log_datapath ${LOG_PATH} \
--labels_path ${LABELS_TXT} \
--datapath ${TRAIN_DATA_PATH} \
--test_val_data ${TEST_DATA_PATH} \
--model_path ${path_to_pre_trained_classifier} >> ${LOG}

# conda deactivate


# echo "* * * * Generating Samples * * * *" >> ${LOG}
# source activate base

# python3 -W ignore \
# /home/toby/MSc_AI/IndividualProject/Conditional-SampleRNN-master/generate_audio.py \
# --exp ${exp_name} \
# --frame_sizes ${FRAME_SIZES} \
# --nb_classes 56 \
# --generate_from ${path_to_generator} \
# --n_samples ${NUM_SAMPLES} \
# --sample_rate 16000 \
# --sample_length 160000 \
# --generate_to ${path_to_results_folder} \
# --n_rnn ${NUM_RNN} >> ${LOG}

# conda deactivate

# source activate pc_base

if find ${path_to_results_folder} -mindepth 1 | read; then
  echo "* * * * Samples Generated * * * *"
else
  echo "Did not generate" >> ${LOG}
  exit
fi


##################### MAKING UPDATED CSV FILE FOR AUGMENTED TRAINING DATA ###########################
python3 write_csv.py \
${BASE}/data/generated_samples/${exp_name}/train_labels_augmented.csv \
${BASE}/data/generated_samples/${exp_name} \
${TRAIN_PATH} \
${TRAIN_DATA_PATH} \
${DATASET}



############## CREATING MELSPECS #######################################

echo "* * * * Creating Melspecs of samples * * * *"
for f in $(find ${path_to_results_folder} -name '*.wav' -or -name '*.mp3')
do
 python3 -W ignore \
 melspectrograms.py \
 ${f} \
 ${BASE}/data/generated_samples/${exp_name}/melspecs/"$(basename ${f} .wav)" \
 --full
done


echo "* * * * Retraining with augmented data * * * * " >> ${LOG}
#Fine-tune on new data
python3 -W ignore \
${BASE}/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/main.py \
train \
${exp_name} \
--train_datapath ${BASE}/data/generated_samples/${exp_name}/train_labels_augmented.csv \
--test_datapath ${TEST_PATH} \
--val_datapath ${VAL_PATH} \
--datapath / \
--log_datapath ${LOG_PATH} \
--test_val_data ${TEST_DATA_PATH} \
--labels_path ${LABELS_TXT} >> ${LOG}


# echo "* * * * Retraining with augmented data * * * * " >> ${LOG}
# echo "* * * * Adversarial Training * * * * *" >> ${LOG}
# #Fine-tune on new data
# python3 -W ignore \
# ${BASE}/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/main.py \
# train_adversarial \
# ${exp_name} \
# --test_datapath ${TEST_PATH} \
# --val_datapath ${VAL_PATH} \
# --log_datapath ${LOG_PATH} \
# --labels_path ${LABELS_TXT} \
# --train_datapath ${BASE}/data/generated_samples/${exp_name}/train_labels_augmented.csv \
# --test_val_data ${TEST_DATA_PATH} \
# --datapath / &> ${LOG}


echo "* * * * Performance with augmented data * * * * " >> ${LOG}
echo "Data: Train split + generated samples" >> ${LOG}
#report performance with augmented data
python3 -W ignore \
${BASE}/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/main.py \
test \
${exp_name} \
--train_datapath ${TRAIN_PATH} \
--test_datapath ${TEST_PATH} \
--val_datapath ${VAL_PATH} \
--log_datapath ${LOG_PATH} \
--labels_path ${LABELS_TXT} \
--datapath ${TRAIN_DATA_PATH} \
--model_path /home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier/${exp_name}.pth \
--test_val_data ${TEST_DATA_PATH} >> ${LOG}

uptime
