import os
import json


out_dir = "archives_data/motionbert_keypoint/predictions"
img_folder_path = "archives_data/frames"

file_list = os.listdir(out_dir)

multi_person_files = []
empty_files = []

for file in file_list:
    # Opening JSON file
    file_path = os.path.join(out_dir, file)
    f = open(file_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    if len(data) > 1:
        multi_person_files.append(file)
    elif len(data) < 1:
        empty_files.append(file)
        
    # keypoints = data[0]["keypoints"]
    # keypoint_scores = data[1]["keypoint_scores"]
    # metainfo = "config/_base_/datasets/coco.py"

    # visualize(os.path.join(img_folder_path, file.replace('.json', '.jpg')), keypoints, keypoint_scores, metainfo=metainfo, show=True)

print("tot_files: ", len(file_list))
print("multi_person_files: ", len(multi_person_files))
print("no_person_detected: ", len(empty_files))

print(multi_person_files)
print(empty_files)