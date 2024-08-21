from mmpose.apis import MMPoseInferencer
from tqdm import tqdm
import os

folder_path = "archives_data/frames"
out_dir = "archives_data/motionbert_keypoint"
augmented_folder_path = "archives_data/frames_augmented"
augmented_out_dir = "archives_data/augmented_motionbert_keypoint"

# # build the inferencer with 3d model alias
# inferencer = MMPoseInferencer(pose3d="human3d")

# # build the inferencer with 3d model config name
# inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

# build the inferencer with 3d model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose3d="configs/body_3d_keypoint/motionbert/h36m/"
    "motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py",
    pose3d_weights="https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/"
    "pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth",
    show_progress=False,
    det_model="yolox_l_8x8_300e_coco",
    det_weights="https://download.openmmlab.com/mmdetection/v2.0/"
    "yolox/yolox_l_8x8_300e_coco/"
    "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    det_cat_ids=[0],  # the category id of 'human' class
)

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(folder_path, out_dir=out_dir)
results = [
    result for result in tqdm(result_generator, total=len(os.listdir(folder_path)))
]

result_generator = inferencer(augmented_folder_path, out_dir=augmented_out_dir)
results = [
    result for result in tqdm(result_generator, total=len(os.listdir(folder_path)))
]

import shutil

# keypoint
dest_folder = "archives_data/keypoints"
shutil.copytree(os.path.join(out_dir, "predictions"), dest_folder)

# keypoint augmented
dest_folder = "archives_data/keypoints_augmented"
shutil.copytree(os.path.join(augmented_out_dir, "predictions"), dest_folder)

list_files_to_rename = os.listdir(dest_folder)
for file in list_files_to_rename:
    old_name = os.path.join(dest_folder, file)
    new_name = os.path.join(dest_folder, 'augmented_' + file)
    os.rename(old_name, new_name)

# keypoint augmented


# results = [result for result in result_generator]

# print("Predictions: ", results)
