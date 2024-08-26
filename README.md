# healtrecsys24

This repository contains the code and data to reproduce the experiments in the paper "Insert Paper Title Here". It includes a pose classification model and tools for extracting keypoint data from human pose estimations.

## Compatibility
This project has been tested on:
- **OS**: Ubuntu 22.04.4 LTS
- **Python**: 3.8.19
- **PyTorch**: 2.1.0

## Installation

1. **Clone the repository**: 
    ```
    git clone https://github.com/GaetanoDibenedetto/healthrecsys24.git
    ```          

2. **MMPose**: 

    Follow the official MMPose installation guidelines [here](https://mmpose.readthedocs.io/en/latest/installation.html). Below is our installation pipeline:: 

    ```
    # We recommend using a virtual environment with Python 3.8
    
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
    pip install -r requirements.txt
    ```     


## Reproducibility of Paper Results
To reproduce the results, download the dataset and our best checkpoint from [Zenodo](https://zenodo.org/records/11075018). Place the downloaded `archives_data` folder in the root directory of the repository, with the following structure:

```
archives_data
│   
├───frames
│
├───keypoints
│       ap_1_250.json
│		...
│		...
│       ms_3_51581.json
│       
├───keypoints_augmented
│       augmented_ap_1_250.json
│		...
│		...
│       augmented_ms_3_51581.json
│       
│───labels
│    └───result
│           labels_for_train.csv
│
└───model_checkpoint
    └───keypoint
            yyyyMMddHHmmss.pkl          
```            

### Execution
Follow these steps to reproduce the results from the paper:

1. **Pose Classification Model (Section 4.2)**:
  - To create your own checkpoint, run the training script for classification:
    - `main_keypoint_classification.py`

2. **Compute Angles**:
  - To compute the average angles based on the keypoint coordinates:
    - `compute_angles.py`

3. **Reproduce the Demo Web App**:
  - To interact with the system via a web app, run the following script:
    - `web_app.py`


## References


