import json
import csv
import numpy as np

from utils.utils import path_pair_file_angles

out_dir = "archives_data/motionbert_keypoint/predictions"
img_folder_path = "archives_data/frames"


json_file = path_pair_file_angles

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


with open(json_file, "r") as file:
    data_dict = json.load(file)

label_file = "archives_data/labels/labels_for_train.csv"

with open(label_file) as file:
    reader = csv.reader(file)
    label_file = list(reader)


correct_frames = []
for file in label_file:
    file_name = file[0]
    label = file[1]

    if label == '[correct_posture]': correct_frames.append(file_name)

print("hello")
