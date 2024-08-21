import csv
import os
import json

import numpy as np

from utils.utils import (
    compute_angles,
    path_pair_file_angles_json,
    path_pair_file_angles_stats_json,
    path_keypoints
)

file_list = os.listdir(path_keypoints)


list_shoulder_level = []
list_hip_angle = []
data_dict = {}

for file in file_list:
    # Opening JSON file
    file_path = os.path.join(path_keypoints, file)
    f = open(file_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    keypoints = data[0]["keypoints"]
    # keypoint_scores = data[0]["keypoint_scores"]

    (
        shoulder_level,
        hip_level,
        y_head_mid_torso,
        y_head_upper_torso,
        y_head_bottom_torso,
        x_head_mid_torso,
        x_head_upper_torso,
        x_head_bottom_torso,
    ) = compute_angles(keypoints)

    data_dict[file] = {
        "shoulders": shoulder_level,
        "hips": hip_level,
        "y_head_bottom_torso": y_head_bottom_torso, 
        "y_head_mid_torso": y_head_mid_torso,
        "y_head_upper_torso": y_head_upper_torso,
        "x_head_bottom_torso": x_head_bottom_torso,
        "x_head_mid_torso": x_head_mid_torso,
        "x_head_upper_torso": x_head_upper_torso,
    }

# save the dictionary to a json file
import json

json_file = path_pair_file_angles_json

with open(json_file, "w") as fp:
    json.dump(data_dict, fp)


# compute the average of each element in the dictionary
shoulders = []
hips = []
x_head_bottom_torso = []
x_head_mid_torso = []
x_head_upper_torso = []
y_head_bottom_torso = []
y_head_mid_torso = []
y_head_upper_torso = []

for k, v in data_dict.items():
    shoulders.append(abs(v["shoulders"]))
    hips.append(abs(v["hips"]))
    x_head_bottom_torso.append(abs(v["x_head_bottom_torso"]))
    x_head_mid_torso.append(abs(v["x_head_mid_torso"]))
    x_head_upper_torso.append(abs(v["x_head_upper_torso"]))
    y_head_bottom_torso.append(abs(v["y_head_bottom_torso"]))
    y_head_mid_torso.append(abs(v["y_head_mid_torso"]))
    y_head_upper_torso.append(abs(v["y_head_upper_torso"]))

def avg(lst):
    return sum(lst) / len(lst)

data_dict_stats = {}
data_dict_stats["avg"] = {
    "shoulders": avg(shoulders),
    "hips": avg(hips),
    "x_head_bottom_torso": avg(x_head_bottom_torso),
    "x_head_mid_torso": avg(x_head_mid_torso),
    "x_head_upper_torso": avg(x_head_upper_torso),
    "y_head_bottom_torso": avg(y_head_bottom_torso),
    "y_head_mid_torso": avg(y_head_mid_torso),
    "y_head_upper_torso": avg(y_head_upper_torso),
}

data_dict_stats["std"] = {
    "shoulders": np.std(shoulders),
    "hips": np.std(hips),
    "x_head_bottom_torso": np.std(x_head_bottom_torso),
    "x_head_mid_torso": np.std(x_head_mid_torso),
    "x_head_upper_torso": np.std(x_head_upper_torso),
    "y_head_bottom_torso": np.std(y_head_bottom_torso),
    "y_head_mid_torso": np.std(y_head_mid_torso),
    "y_head_upper_torso": np.std(y_head_upper_torso),
}

data_dict_stats["max"] = {
    "shoulders": max(shoulders),
    "hips": max(hips),
    "x_head_bottom_torso": max(x_head_bottom_torso),
    "x_head_mid_torso": max(x_head_mid_torso),
    "x_head_upper_torso": max(x_head_upper_torso),
    "y_head_bottom_torso": max(y_head_bottom_torso),
    "y_head_mid_torso": max(y_head_mid_torso),
    "y_head_upper_torso": max(y_head_upper_torso),
}

data_dict_stats["min"] = {
    "shoulders": min(shoulders),
    "hips": min(hips),
    "x_head_bottom_torso": min(x_head_bottom_torso),
    "x_head_mid_torso": min(x_head_mid_torso),
    "x_head_upper_torso": min(x_head_upper_torso),
    "y_head_bottom_torso": min(y_head_bottom_torso),
    "y_head_mid_torso": min(y_head_mid_torso),
    "y_head_upper_torso": min(y_head_upper_torso),
}

# compute the average of correct_postures
label_file = "archives_data/labels/labels_for_train.csv"

with open(label_file) as file:
    reader = csv.reader(file)
    label_file = list(reader)


correct_frames = []
for file in label_file:
    file_name = file[0]
    label = file[1]

    if label == "[correct_posture]":
        correct_frames.append(file_name.replace(".jpg", ""))

correct_shoulders = []
correct_hips = []
correct_x_head_bottom_torso = []
correct_x_head_mid_torso = []
correct_x_head_upper_torso = []
correct_y_head_bottom_torso = []
correct_y_head_mid_torso = []
correct_y_head_upper_torso = []


incorrect_shoulders = []
incorrect_hips = []
incorrect_x_head_bottom_torso = []
incorrect_x_head_mid_torso = []
incorrect_x_head_upper_torso = []
incorrect_y_head_bottom_torso = []
incorrect_y_head_mid_torso = []
incorrect_y_head_upper_torso = []

for k, v in data_dict.items():
    if k.replace('.json', '') in correct_frames:
        correct_shoulders.append(abs(v["shoulders"]))
        correct_hips.append(abs(v["hips"]))
        correct_x_head_bottom_torso.append(abs(v["x_head_bottom_torso"]))
        correct_x_head_mid_torso.append(abs(v["x_head_mid_torso"]))
        correct_x_head_upper_torso.append(abs(v["x_head_upper_torso"]))
        correct_y_head_bottom_torso.append(abs(v["y_head_bottom_torso"]))
        correct_y_head_mid_torso.append(abs(v["y_head_mid_torso"]))
        correct_y_head_upper_torso.append(abs(v["y_head_upper_torso"]))
    else:
        incorrect_shoulders.append(abs(v["shoulders"]))
        incorrect_hips.append(abs(v["hips"]))
        incorrect_x_head_bottom_torso.append(abs(v["x_head_bottom_torso"]))
        incorrect_x_head_mid_torso.append(abs(v["x_head_mid_torso"]))
        incorrect_x_head_upper_torso.append(abs(v["x_head_upper_torso"]))
        incorrect_y_head_bottom_torso.append(abs(v["y_head_bottom_torso"]))
        incorrect_y_head_mid_torso.append(abs(v["y_head_mid_torso"]))
        incorrect_y_head_upper_torso.append(abs(v["y_head_upper_torso"]))


data_dict_stats["correct_avg"] = {
    "shoulders": avg(correct_shoulders),
    "hips": avg(correct_hips),
    "x_head_bottom_torso": avg(correct_x_head_bottom_torso),
    "x_head_mid_torso": avg(correct_x_head_mid_torso),
    "x_head_upper_torso": avg(correct_x_head_upper_torso),
    "y_head_bottom_torso": avg(correct_y_head_bottom_torso),
    "y_head_mid_torso": avg(correct_y_head_mid_torso),
    "y_head_upper_torso": avg(correct_y_head_upper_torso),
}

data_dict_stats["correct_std"] = {
    "shoulders": np.std(correct_shoulders),
    "hips": np.std(correct_hips),
    "x_head_bottom_torso": np.std(correct_x_head_bottom_torso),
    "x_head_mid_torso": np.std(correct_x_head_mid_torso),
    "x_head_upper_torso": np.std(correct_x_head_upper_torso),
    "y_head_bottom_torso": np.std(correct_y_head_bottom_torso),
    "y_head_mid_torso": np.std(correct_y_head_mid_torso),
    "y_head_upper_torso": np.std(correct_y_head_upper_torso),
}

data_dict_stats["correct_max"] = {
    "shoulders": max(correct_shoulders),
    "hips": max(correct_hips),
    "x_head_bottom_torso": max(correct_x_head_bottom_torso),
    "x_head_mid_torso": max(correct_x_head_mid_torso),
    "x_head_upper_torso": max(correct_x_head_upper_torso),
    "y_head_bottom_torso": max(correct_y_head_bottom_torso),
    "y_head_mid_torso": max(correct_y_head_mid_torso),
    "y_head_upper_torso": max(correct_y_head_upper_torso),
}

data_dict_stats["correct_min"] = {
    "shoulders": min(correct_shoulders),
    "hips": min(correct_hips),
    "x_head_bottom_torso": min(correct_x_head_bottom_torso),
    "x_head_mid_torso": min(correct_x_head_mid_torso),
    "x_head_upper_torso": min(correct_x_head_upper_torso),
    "y_head_bottom_torso": min(correct_y_head_bottom_torso),
    "y_head_mid_torso": min(correct_y_head_mid_torso),
    "y_head_upper_torso": min(correct_y_head_upper_torso),
}

data_dict_stats["incorrect_avg"] = {
    "shoulders": avg(incorrect_shoulders),
    "hips": avg(incorrect_hips),
    "x_head_bottom_torso": avg(incorrect_x_head_bottom_torso),
    "x_head_mid_torso": avg(incorrect_x_head_mid_torso),
    "x_head_upper_torso": avg(incorrect_x_head_upper_torso),
    "y_head_bottom_torso": avg(incorrect_y_head_bottom_torso),
    "y_head_mid_torso": avg(incorrect_y_head_mid_torso),
    "y_head_upper_torso": avg(incorrect_y_head_upper_torso),
}

data_dict_stats["incorrect_std"] = {
    "shoulders": np.std(incorrect_shoulders),
    "hips": np.std(incorrect_hips),
    "x_head_bottom_torso": np.std(incorrect_x_head_bottom_torso),
    "x_head_mid_torso": np.std(incorrect_x_head_mid_torso),
    "x_head_upper_torso": np.std(incorrect_x_head_upper_torso),
    "y_head_bottom_torso": np.std(incorrect_y_head_bottom_torso),
    "y_head_mid_torso": np.std(incorrect_y_head_mid_torso),
    "y_head_upper_torso": np.std(incorrect_y_head_upper_torso),
}

data_dict_stats["incorrect_max"] = {
    "shoulders": max(incorrect_shoulders),
    "hips": max(incorrect_hips),
    "x_head_bottom_torso": max(incorrect_x_head_bottom_torso),
    "x_head_mid_torso": max(incorrect_x_head_mid_torso),
    "x_head_upper_torso": max(incorrect_x_head_upper_torso),
    "y_head_bottom_torso": max(incorrect_y_head_bottom_torso),
    "y_head_mid_torso": max(incorrect_y_head_mid_torso),
    "y_head_upper_torso": max(incorrect_y_head_upper_torso),
}

data_dict_stats["incorrect_min"] = {
    "shoulders": min(incorrect_shoulders),
    "hips": min(incorrect_hips),
    "x_head_bottom_torso": min(incorrect_x_head_bottom_torso),
    "x_head_mid_torso": min(incorrect_x_head_mid_torso),
    "x_head_upper_torso": min(incorrect_x_head_upper_torso),
    "y_head_bottom_torso": min(incorrect_y_head_bottom_torso),
    "y_head_mid_torso": min(incorrect_y_head_mid_torso),
    "y_head_upper_torso": min(incorrect_y_head_upper_torso),
}

# save the dictionary to a json file
import json

json_file = path_pair_file_angles_stats_json

with open(json_file, "w") as fp:
    json.dump(data_dict_stats, fp)
    
print("Done!")
print("Files saved to", path_pair_file_angles_json, "and", path_pair_file_angles_stats_json)
