# healtrecsys24

## Compatibility
Tested on Ubuntu 22.04.4 LTS with Python 3.8.19


## Installation

1. **Clone the repository**: 
    ```
    git clone https://github.com/GaetanoDibenedetto/healtrecsys24.git
    ```          


2. **MMPose**: Install MMPose following the official guideline: [LINK](https://mmpose.readthedocs.io/en/latest/installation.html)

    We found trouble with the last version of pythorch.
    We list here our installation pipeline
    ```
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    pip install wheel
    pip install -U openmim
    mim install mmengine
    mim install "mmcv==2.1.0"
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -v -e .
    cd ..
    git clone https://github.com/open-mmlab/mmpose.git
    cd mmpose
    pip install -r requirements.txt
    pip install -v -e .
    cd ..
    ```          

3. **Install our requirements**:
    ```
    pip install -r requireents.txt
    ```     


## Reproducibility of Paper Results
The dataset with our best checkpoint can be downloaded from the following link: [CLICK HERE](https://zenodo.org/records/11075018).

The folders must be inserted in a root folder of the project, after of the git clone of this repository, named `archives_data`, as in the tree structure following below:
```
archives_data
│   
├───frames
│
├───keypoints
│       ap_1_250.jpg.npy
│		...
│		...
│       ms_3_51581.jpg.npy
│       
├───keypoints_augmented
│       augmented_ap_1_250.jpg.npy
│		...
│		...
│       augmented_ms_3_51581.jpg.npy
│       
│───labels
│    └───result
│           labels_for_train.csv
│
└───model_checkpoint
    └───keypoint
            20240803093250.pkl          
```            

### Execution
To reproduce the results mentioned in the associated paper, the following scripts can be utilized in the following order:

1. **Pose Classification Model (Section 4.2)**:
  - `main_keypoint_classification.py`

2. **Compute Angles**:
  - `compute_angles.py`

3. **Reproduce the Demo Web App**:
  - `web_app.py`


### Execution

## References


