# InHome_Physical_Therapy_Pose_Estimation_System
Computer Vision project about In-home physical therapy pose estimation system

# 0. Create the venv, if not done yet:
    python -m venv venv

# 1. Activate the venv:
    .\venv\Scripts\Activate.ps1
    if there is a policy problem, execute this command too
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    then active again the venv


# 2. Install dependancies:
    pip install ultralytics opencv-python scipy torch pyyaml  # webcam acquisition
    pip install scipy torch pyyaml
    pip install fastdtw

# 3. for the NN (in order to train it)
    run generate_quality_labels.py
    run build_windowed_dataset.py
    run train_action_quality_net.py #GRU
    run train_action_quality_tcn.py #TCN

need to put the datasetin a folder on the project root:
project_root/dataset/MMFi_Datasetgit add .