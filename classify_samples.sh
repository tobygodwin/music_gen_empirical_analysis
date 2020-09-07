#!/bin/bash

# Arguments
exp_name="ddsp_model3_default_labels"

LOG=data/generated_samples/${exp_name}/results_classify.txt

#PATHS
BASE="/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework"
# path_to_generator='/home/toby/MSc_AI/IndividualProject/Conditional-SampleRNN-master/results/exp:default_subset-frame_sizes:16,4-n_rnn:2-dataset:mood-theme-default-nb_classes:58/checkpoints/best-ep1-it71375'

######################## EMOTIONAL VALENCE LABELS #######################

# DATASET=emotional_valence
# path_to_pre_trained_classifier=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/output/pretrained_baseline_emotional_valence.pth
# path_to_results_folder=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/audio
# path_to_dataset=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/
# TRAIN_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs
# TEST_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs
# LABELS_TXT=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/emotional_valence.txt
# TRAIN_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_train_labels.csv
# TEST_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_test_labels.csv
# VAL_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/mood_clustering/match_validation_labels.csv
# LOG_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier
# CLASS_MAP=/home/toby/MSc_AI/IndividualProject/Conditional-SampleRNN-master/datasets/mtg-jamendo-emotional-valence-subset/map_class.txt

########################## DEFAULT LABELS ##############################
DATASET=default_labels
path_to_pre_trained_classifier=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/retrain_baseline_default_NEW_AFTERBUG/classifier/retrain_baseline_default_NEW_AFTERBUG.pth
path_to_results_folder=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/audio
path_to_dataset=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/
TRAIN_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs/
TEST_DATA_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/mood-theme/melspecs
LABELS_TXT=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/tags/moodtheme_split.txt
TRAIN_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv
TEST_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv
VAL_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/datasets/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv
LOG_PATH=/home/toby/MSc_AI/IndividualProject/music_gen_evaluation_framework/data/generated_samples/${exp_name}/classifier
CLASS_MAP=/home/toby/MSc_AI/IndividualProject/Conditional-SampleRNN-master/datasets/mood-theme-default/map_class.txt



###################### MAKING UPDATED CSV FILE FOR AUGMENTED TRAINING DATA ###########################
python3 write_csv.py \
${BASE}/data/generated_samples/${exp_name}/test_labels_generated_samples.csv \
${BASE}/data/generated_samples/${exp_name} \
None \
${BASE}/data/generated_samples/${exp_name}/melspecs \
${DATASET}

################ REPORTING PERFORMANCE ON GENERATED SAMPLES  ##########################
echo "* * * * Reporting test statistics * * * * " >> ${LOG}
echo "Training Data: train split" >> ${LOG}
#Running classifier test
python3 -W ignore \
${BASE}/audio_tagging_implementations/mediaeval-2019-moodtheme-detection/submission2/main.py \
test \
${exp_name} \
--train_datapath ${TRAIN_PATH} \
--test_datapath ${BASE}/data/generated_samples/${exp_name}/test_labels_generated_samples.csv \
--val_datapath ${VAL_PATH} \
--log_datapath ${LOG_PATH} \
--labels_path ${LABELS_TXT} \
--datapath ${TRAIN_DATA_PATH} \
--test_val_data ${BASE}/data/generated_samples/${exp_name}/melspecs \
--model_path ${path_to_pre_trained_classifier} >> ${LOG}
uptime
