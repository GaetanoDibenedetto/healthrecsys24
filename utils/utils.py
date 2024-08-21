import json
import os
import random
import numpy as np
import torch
# from skimage import io

path_keypoints = os.path.join("archives_data", "keypoints")

path_keypoints_augmented = os.path.join("archives_data", "keypoints_augmented")

path_frames = os.path.join("archives_data", "frames")

path_frames_augmented = os.path.join("archives_data", "frames_augmented")

path_label = "archives_data/labels/labels_for_train.csv"

path_model_checkpoint = "archives_data/model_checkpoint"

path_pair_file_angles_json = os.path.join("utils", "pair_file_angles.json")

path_pair_file_angles_stats_json = os.path.join("utils", "pair_file_angles_stats.json")

# number_of_keypoints = 17

# get_skelton_info = lambda dataset: joints_dict()[dataset]["skeleton"]
# get_joint_info = lambda dataset: joints_dict()[dataset]["keypoints"]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_keypoint_path(filename):
    if "augmented" in filename:
        keypoint_path = os.path.join(path_keypoints_augmented, filename)
    else:
        keypoint_path = os.path.join(path_keypoints, filename)
    return keypoint_path


def get_frame_path(filename):
    if "augmented" in filename:
        frame_path = os.path.join(path_frames_augmented, filename)
    else:
        frame_path = os.path.join(path_frames, filename)
    frame_path = os.path.join(path_frames, filename)
    return frame_path


def load_keypoint(keypoint_path, normalize_keypoint=True):

    data = json.load(open(keypoint_path))
    keypoint = data[0]['keypoints']

    return load_keypoint_in_torch(keypoint)


def load_keypoint_in_torch(keypoint):
    # removing legs keypoints
    keypoint = np.delete(keypoint, 2)  # first knee
    keypoint = np.delete(keypoint, 5)  # second knee
    keypoint = np.delete(keypoint, 3)  # first feet
    keypoint = np.delete(keypoint, 6)  # second feet

    keypoint = np.array([keypoint])
    keypoint = torch.tensor(keypoint).to(torch.float32)
    return keypoint


# def normalize(volume):
#     """Normalize the volume"""
#     # scale in a 0-1 range
#     volume = (volume - torch.min(volume)) / max(
#         (torch.max(volume) - torch.min(volume)), 1
#     )
#     return volume.to(torch.float32)


def calculate_vector_rotation_x(p1, p2):
    x_axis = [1, 0, 0]
    return calculate_vector_rotation(p1, p2, x_axis)


def calculate_vector_rotation_y(p1, p2):
    y_axis = [0, 0, 1]
    return calculate_vector_rotation(p1, p2, y_axis)


def calculate_vector_rotation(p1, p2, axis=[1, 0, 0]):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    def compute_unit_vector(vector):
        # Calculate the magnitude (length) of the vector
        magnitude = np.linalg.norm(vector)

        # Normalize the vector (unit vector)
        unit_vector = vector / magnitude

        return unit_vector

    # Calculate the vector
    vector = p2 - p1

    # Calculate the angle with respect to the x-axis
    axis = np.array(axis)

    vector = compute_unit_vector(vector)
    axis = compute_unit_vector(axis)

    # Dot product of the unit vector and the x-axis
    dot_product = np.dot(vector, axis)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    # anti-clockwise
    if angle_degrees > 90:
        angle_degrees = angle_degrees - 180

    return angle_degrees

def compute_angles(keypoints):
    # shoulders keypoint
    point1 = keypoints[14]
    point2 = keypoints[11]

    shoulder_level = calculate_vector_rotation_x(point1, point2)

    # hips keypoint
    point1 = keypoints[1]
    point2 = keypoints[4]

    hip_level = calculate_vector_rotation_x(point1, point2)

    # head direction respect bottom torso, respect mid torso and respect to upper torso
    head = keypoints[10]
    bottom_torso = keypoints[0]
    mid_torso = keypoints[7]
    upper_torso = keypoints[8]

    y_head_bottom_torso = calculate_vector_rotation_y(head, bottom_torso)
    y_head_mid_torso = calculate_vector_rotation_y(head, mid_torso)
    y_head_upper_torso = calculate_vector_rotation_y(head, upper_torso)
    
    x_head_bottom_torso = calculate_vector_rotation_x(head, bottom_torso)
    x_head_mid_torso = calculate_vector_rotation_x(head, mid_torso)
    x_head_upper_torso = calculate_vector_rotation_x(head, upper_torso)

    return shoulder_level, hip_level, y_head_mid_torso, y_head_upper_torso, y_head_bottom_torso, x_head_mid_torso, x_head_upper_torso, x_head_bottom_torso
