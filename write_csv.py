import os
import csv
import shutil
import argparse

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("csv_path", help="Path to csv")
    parser.add_argument("sample_path", help="Path to generated samples")
    parser.add_argument("path_to_old_csv", help="path to existing csv")
    parser.add_argument("path_to_training_data", help="path to training melspecs")
    parser.add_argument("dataset", help="Default or emotional_valence")
    # Parse arguments
    args = parser.parse_args()

    return args


def write_csv(path_csv,names_labels,path_to_generated_samples=None):
    data = []

    for file, label in names_labels.items():
        if path_to_generated_samples:
            if dataset == 'default_labels':
                row = {
                    'TRACK_ID': 'Darude' ,
                    'ARTIST_ID': 'sandstorm',
                    'ALBUM_ID': 'best_ever',
                    'PATH': path_to_generated_samples + '/melspecs/' + file,
                    'DURATION': '234',
                    'TAGS': 'mood/theme---' + label
                }
            elif dataset == 'emotional_valence':
                row = {
                    'TRACK': path_to_generated_samples + '/melspecs/' + file,
                    'LABEL': '',
                    'DURATION': '',
                    'BIN': '',
                    'EMOTION_VALENCE': label
                }
        else:
            if dataset == 'default_labels':
                row = {
                'TRACK_ID': '1234' ,
                'ARTIST_ID': 'toby',
                'ALBUM_ID': 'godwin',
                'PATH': os.path.dirname(file)[:-2] + os.path.basename(file),
                'DURATION': '234',
                'TAGS':'mood/theme---'+ label
            }
            elif dataset == 'emotional_valence':
                row = {
                    'TRACK': file,
                    'LABEL': '',
                    'DURATION': '',
                    'BIN': '',
                    'EMOTION_VALENCE': label
                }
        data.append(row)

    if dataset == 'default_labels':
        fieldnames = ['TRACK_ID', 'ARTIST_ID', 'ALBUM_ID', 'PATH', 'DURATION', 'TAGS']
    elif dataset == 'emotional_valence':
        fieldnames = ['TRACK', 'LABEL', 'DURATION', 'BIN', 'EMOTION_VALENCE']

    if path_to_generated_samples:
        with open(path_csv, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=fieldnames, delimiter='\t')
            #writer.writeheader()
            writer.writerows(data)
    else:
        with open(path_csv, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(data)


def copy_csv(path_old_csv,path_to_csv):
    source = path_old_csv
    destination = path_to_csv
    shutil.copyfile(source, destination)

def update_paths(path_to_old_csv,path_to_csv, path_to_melspecs):
    
    with open(path_to_old_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        rows = list(csv_reader)

    lst_idx_original_files = []
    for idx,row in enumerate(rows):
        if idx>0: #afer header
            if dataset == 'default_labels':
                name_idx = 3
                label_idx = 5
                row[name_idx] = path_to_melspecs + row[name_idx][3:]
            elif dataset == 'emotional_valence':
                name_idx = 0
                row[name_idx] = path_to_melspecs + row[name_idx]

    with open(path_to_csv, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerows(rows)


def get_names_labels(path_to_samples):
    # path_to_samples = os.path.join(path_to_samples,'audio')
    names_labels=dict()
    labels = [x[0] for x in os.walk(path_to_samples)]
    for label in labels: #looping through sub dirs
        list_ = os.listdir(os.path.join(path_to_samples,label)) #list files in subdir
        for file_ in list_: #loop through files in sub dir
            name, ext = os.path.splitext(file_)

            if ext != '.wav':
                continue
            names_labels[file_] = os.path.basename(label)
    return names_labels


if __name__=="__main__":
    args = parseArguments()
    path_to_csv = args.csv_path
    path_to_old_csv = args.path_to_old_csv
    path_to_samples = args.sample_path
    path_to_melspecs = args.path_to_training_data
    dataset = args.dataset
    if path_to_old_csv != 'None': #if retrain
        copy_csv(path_to_old_csv, path_to_csv)

        update_paths(path_to_old_csv, path_to_csv, path_to_melspecs)
        names_labels = get_names_labels(path_to_samples)

        write_csv(path_to_csv, names_labels,path_to_generated_samples=path_to_samples)

    else:
        names_labels = get_names_labels(path_to_samples)

        write_csv(path_to_csv, names_labels)
