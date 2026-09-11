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

# InHome Physical Therapy Pose Estimation System

A real-time pose estimation system for home-based physical therapy. The system evaluates the execution quality of six rehabilitation exercises using a standard webcam, without requiring wearable sensors.

For each exercise, the patient's 2D body pose is extracted from the video stream using YOLOv8-Pose and compared against reference motion patterns derived from the **MM-Fi** dataset. Two complementary approaches are implemented:

1. **Geometric rule-based approach** — evaluates the movement using joint angles or other pose-derived signals and compares them against exercise-specific reference patterns.
2. **Neural approach** — uses either a GRU or a TCN to process normalized pose sequences and jointly predict the performed exercise and an execution-quality score ranging from 0 to 100.

The system provides a real-time quality score for each detected repetition and can optionally record the results in CSV log files.

> Project developed for the *Industrial Applications of Computer Vision* course (A.Y. 2025–26).

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [How to Run the Project](#how-to-run-the-project)
6. [System Input and Output](#system-input-and-output)
7. [MM-Fi Dataset Evaluation Pipeline](#mm-fi-dataset-evaluation-pipeline)
8. [Neural Pipeline (GRU vs TCN)](#neural-pipeline-gru-vs-tcn)
9. [Metrics and Baselines](#metrics-and-baselines)
10. [Third-Party Code and Libraries](#third-party-code-and-libraries)
11. [AI/LLM Tool Usage](#aillm-tool-usage)
12. [References](#references)
13. [Known Limitations](#known-limitations)

---

## Overview

The system addresses the problem of evaluating **how correctly a patient performs a physical therapy exercise in front of a home webcam**, without relying on wearable sensors.

- **Input**: RGB video stream from a webcam.
- **Pose extraction**: YOLOv8-Pose (`yolov8n-pose.pt`, Ultralytics) detects 17 COCO keypoints for each frame.
- **Evaluation**: two parallel and comparable approaches are implemented, both calibrated or trained using the same MM-Fi dataset:
  1. **Rule-based (geometric)**: joint angles or other pose-derived signals are compared against an exercise-specific reference curve extracted from the dataset. A state machine is used to detect individual repetitions.
  2. **Neural**: a GRU or TCN processes windows of normalized keypoints to simultaneously predict the performed exercise (classification) and an execution-quality score from 0 to 100 (regression). The training target is a weak label generated from the same heuristic score used by the rule-based pipeline.
- **Output**: an execution-quality score from 0 to 100% for each repetition or analysis window, displayed as a video overlay, with optional CSV logging in `logs/`.

The system currently supports the following exercises:

| Exercise | MM-Fi Action | Measured Signal |
|---|---|---|
| Squat | A12 | Left knee angle (hip–knee–ankle) |
| Left lunge | A15 | Left knee angle |
| Right lunge | A16 | Right knee angle |
| Left limb extension | A07 | Left elbow angle (shoulder–elbow–wrist) |
| Right limb extension | A08 | Right elbow angle |
| Jumping jack | A26 (proxy: "jumping up"; MM-Fi has no dedicated action) | Leg opening (ankle-to-shoulder ratio) + arm elevation |

---

## Repository Structure

```text
InHome_Physical_Therapy_Pose_Estimation_System/
│
├── mmfi_lib/                          # Official MM-Fi dataset toolkit (third-party, adapted)
│   ├── mmfi.py                        #   MMFi_Database / MMFi_Dataset / DataLoader
│   └── evaluate.py                    #   Original MM-Fi metrics (MPJPE, PA-MPJPE)
│
├── src/
│   ├── utils.py                       # COCO keypoint indices, project paths,
│   │                                  # calculate_angle(), calculate_depth_score(),
│   │                                  # keypoints_are_valid(), select_patient_keypoints()
│   ├── main.py                        # Tkinter GUI: launches reference extraction
│   │                                  # and real-time comparison for the selected exercise
│   ├── compare_models.py              # GRU vs TCN comparison (accuracy, MAE, CPU latency)
│   │
│   ├── reference_extraction/          # Builds reference curves from MM-Fi
│   │   ├── reference_extraction_squat.py
│   │   ├── reference_extraction_lunge.py              # both sides, A15/A16
│   │   ├── reference_extraction_limb_extensions.py    # both sides, A07/A08
│   │   └── reference_extraction_jumping_jacks.py      # A26
│   │
│   ├── evaluate_dataset/              # Offline analysis of the MM-Fi dataset:
│   │   ├── evaluate_dataset_fixed_targets_squat.py
│   │   ├── evaluate_dataset_fixed_targets_lunge_left.py
│   │   ├── evaluate_dataset_fixed_targets_lunge_right.py
│   │   ├── evaluate_dataset_fixed_targets_limb_extension_left.py
│   │   ├── evaluate_dataset_fixed_targets_limb_extension_right.py
│   │   └── evaluate_dataset_fixed_targets_jumping_jacks.py
│   │
│   ├── realtime_comparison/           # Live webcam inference
│   │   ├── realtime_comparison_squat.py
│   │   ├── realtime_comparison_lunge_movements.py
│   │   ├── realtime_comparison_limb_extensions.py
│   │   ├── realtime_comparison_jumping_jacks.py
│   │   └── realtime_inference_action_quality.py   # GRU/TCN-based variant
│   │
│   └── neural_network/                # Quality-model training pipeline
│       ├── keypoint_normalize.py      # normalization (hip centering, shoulder scale)
│       │                              # + mirroring shared by training and live inference
│       ├── generate_quality_labels.py # step 1: quality labels per subject/action
│       ├── build_windowed_dataset.py  # step 2: sliding-window dataset (.npz)
│       ├── train_action_quality_net.py# step 3a: GRU training
│       └── train_action_quality_tcn.py# step 3b: TCN training
│
├── logs/                               # Session CSV logs (one row per repetition)
├── references/  (generated, git-ignored) # Reference curves (.npy) for each exercise
├── GRU/ , TCN/  (generated, git-ignored)  # .npz datasets and model checkpoints
├── dataset/MMFi_Dataset/ (NOT included)  # MM-Fi dataset downloaded manually
├── yolov8n-pose.pt (NOT included)        # YOLOv8-Pose weights
│
├── requirements / pip commands         # See Installation
├── LICENSE                              # Apache License 2.0
└── README.md
```

### Main Modules

- **`mmfi_lib/`**: provides the interface used to load the MM-Fi dataset through PyTorch `Dataset`/`DataLoader` objects. It is not original project code; it is derived from the official MM-Fi toolkit associated with the paper, reused to read the dataset's RGB 2D keypoint sequences.
- **`src/utils.py`**: centralizes the geometry and scoring functions so that offline evaluation on MM-Fi and live webcam evaluation use the same logic.
- **`src/reference_extraction/`**: loads one or more MM-Fi sequences for each exercise, extracts the relevant signal or signals, resamples them to a fixed length, and stores the resulting average reference curve in `references/`.
- **`src/evaluate_dataset/`**: evaluates all subjects in an MM-Fi environment against an exercise target. The squat uses a fixed target, while the other exercises use empirical targets estimated from the population because no published clinical target is available for them. The scripts also help identify subjects that can be used as reference examples.
- **`src/realtime_comparison/`**: captures webcam frames, runs YOLOv8-Pose frame by frame, tracks the active patient using `select_patient_keypoints`, calibrates thresholds from the patient's observed range of motion during the first few seconds, detects repetitions through a state machine, and assigns a score to each repetition.
- **`src/neural_network/`**: provides a separate alternative pipeline based on a GRU or TCN. Instead of relying on fixed joint-angle rules, it learns from normalized keypoint windows to predict both the exercise and a 0–100 quality score using the heuristic score generated by the rule-based pipeline as a weak training label.
- **`src/main.py`**: provides a Tkinter graphical interface that allows the user to select an exercise and then runs the corresponding reference-extraction and real-time comparison steps.

---

## Dataset

The project uses **MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset**, presented at NeurIPS 2023 in the Datasets and Benchmarks Track. The dataset contains 40 subjects performing 27 daily and rehabilitation actions across five synchronized modalities: RGB, depth, LiDAR, mmWave, and WiFi-CSI.

This project uses **only the RGB modality**, specifically the annotated 2D keypoints, as the reference representation against which YOLOv8-Pose estimates from the webcam stream are compared.

- Project page: https://ntu-aiot-lab.github.io/mm-fi
- Official toolkit/repository: https://github.com/ybhbingo/MMFi_dataset
- Paper: Yang et al., *"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing"*, NeurIPS 2023, arXiv:2305.10345.

### Dataset Setup

The dataset is **not included in this repository** because of its size and is excluded from Git through `.gitignore`.

Download it using the links provided by the official MM-Fi repository and place it in the following location:

```text
<project root>/dataset/MMFi_Dataset/
    E01/S01/A01/rgb/frame001.npy ...
    E01/S01/A01/ground_truth.npy
    ...
```

`src/utils.py` automatically defines:

```text
DATASET_ROOT = <PROJECT_ROOT>/dataset/MMFi_Dataset
```

---

## Installation

Create and activate a Python virtual environment, then install the required dependencies:

```bash
# 0. Create the virtual environment if it does not already exist
python -m venv venv

# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# If PowerShell reports an execution-policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then activate the virtual environment again

# 2. Install the dependencies
pip install ultralytics opencv-python scipy torch pyyaml fastdtw
```

### Main Dependencies

- `torch` — training and inference of the GRU/TCN models.
- `ultralytics` — YOLOv8-Pose inference.
- `opencv-python` — webcam acquisition and video/image processing.
- `scipy` — reading `.mat` files used by parts of the MM-Fi toolkit; WiFi-CSI is not used by the active pipeline, but the dependency is required by `mmfi_lib`.
- `numpy` — numerical processing and storage of intermediate data.
- `tkinter` — graphical user interface; included with standard Python installations on Windows.

The `yolov8n-pose.pt` weights are downloaded automatically by Ultralytics on first use. They can also be placed manually in the project root.

---

## How to Run the Project

### Option A — Graphical Interface (Recommended)

```bash
python src/main.py
```

A window opens with one button for each supported exercise. When an exercise is selected, the interface automatically runs the corresponding `reference_extraction_*.py` script when necessary and then launches the matching real-time comparison script from `realtime_comparison/`.

### Option B — Run an Exercise Manually

First generate the reference curve, then start the real-time webcam comparison:

```bash
# 1. Build the reference curve for the selected exercise
python src/reference_extraction/reference_extraction_squat.py

# 2. Start real-time webcam comparison
python src/realtime_comparison/realtime_comparison_squat.py
```

The same procedure applies to:

- `reference_extraction_lunge.py` / `realtime_comparison_lunge_movements.py`
- `reference_extraction_limb_extensions.py` / `realtime_comparison_limb_extensions.py`
- `reference_extraction_jumping_jacks.py` / `realtime_comparison_jumping_jacks.py`

### Main Parameters

The following parameters can be modified at the beginning of the relevant scripts or in `src/utils.py`:

| Parameter | Location | Description |
|---|---|---|
| `SUBJECTS` | `reference_extraction_*.py` | MM-Fi subjects used to build the reference |
| `DEPTH_TOLERANCE`, `DEPTH_FALLOFF_RANGE` | `utils.py` | Width of the acceptable region and score falloff range |
| `KP_CONF_THRESHOLD` | `utils.py` | Minimum YOLO confidence required for a keypoint to be considered valid |
| `USE_ADAPTIVE_THRESHOLDS`, `CALIBRATION_DURATION` | `realtime_comparison_*.py` | Enables threshold calibration based on the patient's observed range of motion |
| `CAMERA_INDEX` | `realtime_comparison_*.py` | Webcam index |
| `TRACKED_SIDE` | Bilateral exercise scripts | `'auto'`, `'left'`, or `'right'` |

### Option C — Offline Evaluation on the MM-Fi Dataset

```bash
python src/evaluate_dataset/evaluate_dataset_fixed_targets_squat.py
# One evaluation script is available for each exercise in src/evaluate_dataset/
```

The scripts calculate a score for each subject in environment `E01` and print a summary including the mean, standard deviation, and number of subjects within the selected tolerance.

### Option D — Neural Pipeline (GRU/TCN)

See the [Neural Pipeline](#neural-pipeline-gru-vs-tcn) section for the complete execution order.

---

## System Input and Output

### Input

- Local webcam stream using a configurable camera index, with the device's default resolution and frame rate.
- For offline processing, RGB sequences from MM-Fi (`.npy` files containing 17 COCO keypoints × `(x, y)`).

### Output

- Real-time video overlay showing the current state of the state machine (calibration, movement, or rest), the current repetition quality score from 0 to 100%, and calibration/warning messages.
- A terminal log entry for each completed repetition, for example:

```text
Rep: depth_achieved=... score=...%
```

- A session CSV file in `logs/` containing, for each repetition:
  - timestamp
  - achieved depth or measured signal
  - target
  - difference
  - calibration offset
  - percentage score
  - number of frames in the repetition

  An example is `logs/session_20260901_133110.csv`.

- For the neural pipeline: the predicted exercise and quality score from 0 to 100, updated approximately once per second during live inference.

### Explicit Scope

The system does **not** provide a clinical diagnosis, does not replace assessment by a physiotherapist, and does not guarantee correct recognition when multiple people are present in the scene. The implementation includes a tracking mechanism designed to keep the closest/most central person as the active patient.

---

## MM-Fi Dataset Evaluation Pipeline

For each exercise, the evaluation workflow is:

1. **`evaluate_dataset_fixed_targets_*.py`** — loads the subjects from an MM-Fi environment, computes the relevant signal or signals, estimates an exercise target, and calculates a subject-level score using the same `calculate_depth_score()` function used during live inference. The squat uses a fixed target of 95°, while the other exercises use empirical population-based targets because no published clinical reference is available. The scripts also identify subjects within the selected tolerance who can be considered as reference candidates.
2. **`reference_extraction_*.py`** — takes one or more selected MM-Fi subjects, resamples their sequences to a fixed length of 100 points, and averages them into a single reference curve stored in `references/`.
3. **`realtime_comparison_*.py`** — loads the reference curve, calibrates the thresholds for the individual patient (or inherits the reference thresholds when adaptive calibration is disabled), and evaluates each repetition in real time using the same `calculate_depth_score()` function.

This design ensures that **offline dataset evaluation and online webcam evaluation remain numerically comparable**, because both share the same implementation in `src/utils.py`.

---

## Neural Pipeline (GRU vs TCN)

The neural pipeline is an alternative and complementary approach to the rule-based method. It is designed to compare an explicit geometric strategy with a data-driven model.

Run the scripts in the following order:

```bash
python src/neural_network/generate_quality_labels.py    # -> GRU/quality_labels.csv, TCN/quality_labels.csv
python src/neural_network/build_windowed_dataset.py      # -> GRU/action_quality_dataset.npz, TCN/action_quality_dataset.npz
python src/neural_network/train_action_quality_net.py    # -> GRU/action_quality_net.pt
python src/neural_network/train_action_quality_tcn.py    # -> TCN/action_quality_tcn.pt
python src/compare_models.py                              # GRU vs TCN comparison
python src/realtime_comparison/realtime_inference_action_quality.py   # live inference
```

### Pipeline Steps

- **`generate_quality_labels.py`** recalculates the same heuristic quality score for all 40 subjects across environments E01–E04. It does not introduce a new definition of correct exercise execution.
- **`build_windowed_dataset.py`** normalizes each sequence by centering it on the mid-hip and scaling it according to shoulder width, as implemented in `keypoint_normalize.py`. The sequences are divided into sliding windows of 30 frames with a stride of 10. Left/right mirroring is also applied as data augmentation, so a mirrored left-lunge sample can be used as a synthetic right-lunge sample. The train/validation split is performed **by subject rather than by window**, ensuring that validation includes subjects not seen during training.
- **`train_action_quality_net.py`** and **`train_action_quality_tcn.py`** use the same overall architecture: a shared encoder followed by two output heads, one for exercise classification and one for quality-score regression. They use the same loss, `CrossEntropy + 0.5 × MSE`, and the same class weights. The only architectural difference is the encoder: recurrent GRU versus convolutional TCN.
- **`compare_models.py`** evaluates both checkpoints on the same validation split and reports classification accuracy, score MAE, per-class accuracy, parameter count, and CPU inference latency. CPU latency is particularly relevant because the intended application is real-time inference on a relatively low-performance computer.

### Important Training Limitation

The neural network does **not** learn a clinical definition of correct exercise form. Instead, it learns to reproduce the same population-relative heuristic score already used by the rule-based pipeline.

The quality target is therefore a **weak label**, not a clinically validated ground-truth score.

---

## Metrics and Baselines

- **Quality score metric**: the two-zone `calculate_depth_score()` function assigns scores from 100% to 80% linearly within `DEPTH_TOLERANCE`, then decreases from 80% to 0% across `DEPTH_FALLOFF_RANGE`. This is an internal project metric used consistently in both offline and online evaluation.
- **Baseline**: the squat uses a fixed target of 95° as a simple baseline based on the clinical value used by the project. For the other exercises, where no published clinical target is available, the baseline is the empirical target calculated as the mean of the MM-Fi population by `evaluate_dataset_fixed_targets_*.py`.
- **Model comparison**: GRU and TCN are compared using classification accuracy, quality-score MAE on the 0–100 scale, parameter count, and CPU latency per window through `compare_models.py`. The neural pipeline can also be compared with the rule-based pipeline on the same subjects and actions, providing a comparison between an explicit geometric method and a deep-learning method.
- **Pose-estimation metrics**: `mmfi_lib/evaluate.py` also provides the standard MPJPE and PA-MPJPE metrics from the original MM-Fi toolkit, including Procrustes alignment, for a possible direct evaluation of pose-estimation quality against the dataset's 3D ground truth.

---

## Third-Party Code and Libraries

| Component | Source | License | Role in the Project |
|---|---|---|---|
| `mmfi_lib/mmfi.py`, `mmfi_lib/evaluate.py` | Official MM-Fi toolkit (https://github.com/ybhbingo/MMFi_dataset) | See original repository | Dataset loading, PyTorch `Dataset`/`DataLoader`, MPJPE/PA-MPJPE metrics |
| YOLOv8-Pose (`yolov8n-pose.pt`) | Ultralytics (https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Real-time 2D keypoint estimation from the webcam stream |
| PyTorch | https://pytorch.org/ | BSD-style | GRU/TCN training and inference |
| OpenCV (`opencv-python`) | https://opencv.org/ | Apache 2.0 | Video acquisition and visualization, image/depth processing |
| SciPy | https://scipy.org/ | BSD | Reading `.mat` files used by the MM-Fi toolkit |
| This project | — | Apache License 2.0 | See `LICENSE` |

---

## AI/LLM Tool Usage

The course guidelines require the use of AI tools to be explicitly documented when they contribute to the project.

This README was prepared with the assistance of an AI/LLM assistant (Claude, Anthropic), based on the project source code and the course guidelines provided by the authors. The technical content is intended to reflect the implementation present in the repository at the time of writing.

If additional AI assistance was used for code generation, refactoring, debugging, documentation, or other tasks, those contributions should be described here together with the parts that were manually written, checked, and validated by the authors.

---

## References

- J. Yang, H. Huang, Y. Zhou, X. Chen, Y. Xu, S. Yuan, H. Zou, C. X. Lu, L. Xie. *"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing"*. NeurIPS 2023, Datasets and Benchmarks Track. [arXiv:2305.10345](https://arxiv.org/abs/2305.10345) · [Project page](https://ntu-aiot-lab.github.io/mm-fi) · [Repository/toolkit](https://github.com/ybhbingo/MMFi_dataset)
- Ultralytics YOLOv8 (pose estimation). [Documentation](https://docs.ultralytics.com/tasks/pose/) · [Repository](https://github.com/ultralytics/ultralytics)
- Procrustes alignment / PA-MPJPE: implementation adapted in `mmfi_lib/evaluate.py` from a common MATLAB `procrustes` implementation ported to NumPy, following a standard approach in 3D human pose estimation.

Additional papers or solutions consulted for the GRU/TCN architecture or metric design should be added here if they were used as references beyond the project source code.

---

## Known Limitations

- MM-Fi does not include a dedicated **jumping jack** action. The project therefore uses **A26 ("jumping up")** as the closest available proxy. Any target or score produced for this exercise should be considered an approximation rather than a validated reference.
- The targets used for lunges and limb extensions are **empirical estimates** based on the mean of the MM-Fi population. They are not clinically validated targets, unlike the squat, which uses a fixed 95° target based on the project's clinical reference.
- The **GRU/TCN neural network does not learn a clinical definition of correct form**. It learns to reproduce the same heuristic score generated by the rule-based pipeline.
- The system assumes a single active patient in the scene. It includes tracking intended to handle temporary additional people, such as a caregiver, but reliable pose estimation still requires a front-facing webcam with the subject fully visible.
