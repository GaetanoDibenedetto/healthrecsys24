import os
import pandas as pd
import re
import os
import numpy as np

class LoadDatasetKeypoints():
    def __init__(self, keypoint_path = '', dataframes_path = '', make_framing_from_video = True, pad_lenght_wanted = 1, action_to_remove = []):
        self.keypoint_path = keypoint_path

        pass

    def extract_info_from_filename(self, filename):
        match = re.match(r"([a-zA-Z]+)_(\d+)_(\d+)\.json", filename)
        if match:
            frame_name = filename
            subject = match.group(1)
            clip_number = int(match.group(2))
            frame_number = int(match.group(3))
            return {
                "path": frame_name,
                "augmented": False,
                "subject": subject,
                "clip_number": clip_number,
                "frame_number": frame_number,
            }
        else:
            match = re.match(r"([a-zA-Z]+)_([a-zA-Z]+)_(\d+)_(\d+)\.json", filename)
            if match:
                frame_name = filename
                augmented = True
                subject = match.group(2)
                clip_number = int(match.group(3))
                frame_number = int(match.group(4))
                return {
                    "path": frame_name,
                    "augmented": augmented,
                    "subject": subject,
                    "clip_number": clip_number,
                    "frame_number": frame_number,
                }
            else:
                return None

    def load_dataset_info(self,
        keypoint_path, label_path=None):

        if label_path!=None:
            self.label_path = label_path
            self.df_label = pd.read_csv(label_path)

        files_list = os.listdir(keypoint_path)
        files_list = [x for x in files_list if x.endswith(".json")]

        path, augmented, subject, clip_number, frame_number, label = [], [], [], [], [], []
        for file in files_list:
            file_info = self.extract_info_from_filename(file)
            if file_info:
                # to fix
                try:
                    label.append(
                        self.df_label[self.df_label["filename"] == file_info["path"].replace('augmented_', '').replace(".json",".jpg")]["label"].item()
                    )
                except:
                    continue
                                    
                path.append(file_info["path"])
                augmented.append(file_info["augmented"])
                subject.append(file_info["subject"])
                clip_number.append(file_info["clip_number"])
                frame_number.append(file_info["frame_number"])
            else:
                raise ValueError("Filename format doesn't match.")

        data = {
            "path": path,
            "augmented": augmented,
            "subject": subject,
            "clip_number": clip_number,
            "frame_number": frame_number,
            "label": label,
        }
        df = pd.DataFrame(data)

        return df

def split_dataset(df, random_state=42):
    # non voglio che i soggetti nel training set vadano nel test set
    test_subjects = {"gd"}
    subjects = set(df["subject"].unique())
    train_subjects = subjects - test_subjects

    len(train_subjects), len(test_subjects)

    train_data = df[df["subject"].isin(train_subjects)]
    test_data = df[df["subject"].isin(test_subjects)]

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return train_data, test_data


def balance_dataset(df):
    # remove the label that ecced the counter of the lower one
    label_count = df["label_id"].value_counts()
    min_label_count = label_count.min()
    df.loc[:, "label_id_str"] = df["label_id"].astype(str)
    df = df.groupby("label_id_str").sample(min_label_count)
    label_count = df["label_id"].value_counts()
    return df

def map_label(df):
    label_list = df["label"].explode().unique()
    mapping = onehotencoder(label_list)

    df["label_id"] = df["label"].explode().map(mapping)
    return df, label_list

def onehotencoder(unique_list_labels):
    unique_list_labels.sort()
    mapping = {x: i for i, x in enumerate(unique_list_labels)}

    for key, value in mapping.items():
        mapping[key] = np.zeros(len(unique_list_labels))
        mapping[key][value] = 1

    return mapping       
