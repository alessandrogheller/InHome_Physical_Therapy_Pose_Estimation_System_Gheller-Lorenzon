# Rehab Posture Correction

A pose-estimation system for home-based physical therapy: it evaluates, in real time via webcam, the execution quality of six rehabilitation exercises by comparing the patient's 2D skeleton (extracted with YOLOv8-Pose) against reference curves built on the **MM-Fi** dataset, using both a geometric approach based on joint angles and a neural network (GRU/TCN) trained on the same data.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Evaluation Dataset](#evaluation-dataset)
5. [Ground Truth](#ground-truth)
6. [Installation](#installation)
7. [How to Run the Project](#how-to-run-the-project)
8. [Camera and Scene Setup](#camera-and-scene-setup)
9. [System Input and Output](#system-input-and-output)
10. [MM-Fi Dataset Evaluation Pipeline](#mm-fi-dataset-evaluation-pipeline)
11. [Neural Pipeline (GRU vs TCN)](#neural-pipeline-gru-vs-tcn)
12. [Metrics and Baselines](#metrics-and-baselines)
13. [Third-Party Code and Libraries](#third-party-code-and-libraries)
14. [AI/LLM Tool Usage](#aillm-tool-usage)
15. [References](#references)
16. [Known Limitations](#known-limitations)

---

## Overview

The system addresses the problem of evaluating **how correctly a patient performs a physical therapy exercise in front of a home webcam**, without relying on wearable sensors.

- **Input**: RGB video stream from a webcam.
- **Pose extraction**: YOLOv8-Pose (`yolov8n-pose.pt`, Ultralytics) detects 17 COCO keypoints for each frame.
- **Evaluation**: two parallel and comparable approaches are implemented, both calibrated or trained using the same MM-Fi dataset:
  1. **Rule-based (geometric)**: joint angles or other pose-derived signals (e.g. ankle-to-ankle distance) are compared against an exercise-specific reference curve extracted from the dataset. A state machine is used to detect individual repetitions.
  2. **Neural**: a GRU or TCN processes windows of normalized keypoints to simultaneously predict the performed exercise (classification) and an execution-quality score from 0 to 100 (regression). The training target is a weak label generated from the same heuristic score used by the rule-based pipeline.
- **Output**: an execution-quality score from 0 to 100% for each repetition or analysis window, displayed as a video overlay.

The system currently supports the following exercises:

| Exercise | MM-Fi Action | Measured Signal |
|---|---|---|
| Squat | A12 | Left knee angle (hip–knee–ankle) |
| Left lunge | A15 | Left knee angle |
| Right lunge | A16 | Right knee angle |
| Left limb extension | A07 | Left elbow angle (shoulder–elbow–wrist) |
| Right limb extension | A08 | Right elbow angle |
| Jumping jack | A26  | Leg opening (ankle-to-shoulder ratio) + arm elevation |

---

## Repository Structure

```
Rehab Posture Correction/
│
├── mmfi_lib/                              # Official MM-Fi dataset toolkit (third-party, adapted)
│   ├── mmfi.py                            #   MMFi_Database / MMFi_Dataset / DataLoader
│   └── evaluate.py                        #   Original MM-Fi metrics (MPJPE, PA-MPJPE)
│
├── src/
│   ├── utils.py                           # COCO keypoint indices, project paths,
│   │                                       # calculate_angle(), calculate_depth_score(),
│   │                                       # keypoints_are_valid(), select_patient_keypoints()
│   ├── main.py                            # Tkinter GUI: launches reference extraction
│   │                                       # and real-time comparison for the selected exercise
│   │
│   ├── reference_extraction/              # Builds reference curves from MM-Fi
│   │   ├── reference_extraction_squat.py
│   │   ├── reference_extraction_lunge.py              (both sides, A15/A16)
│   │   ├── reference_extraction_limb_extensions.py    (both sides, A07/A08)
│   │   └── reference_extraction_jumping_jacks.py      (A26)
│   │
│   ├── evaluate_dataset/                  # Offline analysis of the MM-Fi dataset:
|   |   ├── build_eval_windowed_dataset.py                     # sliding-window evaluates dataset 
│   │   ├── evaluate_dataset_fixed_targets_squat.py            # computes targets/tolerances
│   │   ├── evaluate_dataset_fixed_targets_lunge_left.py       # and a per-subject score, used
│   │   ├── evaluate_dataset_fixed_targets_lunge_right.py      # as a baseline and to pick
│   │   ├── evaluate_dataset_fixed_targets_limb_extension_left.py   # "good" subjects to use
│   │   ├── evaluate_dataset_fixed_targets_limb_extension_right.py  # as reference
│   │   ├── evaluate_dataset_fixed_targets_jumping_jacks.py
|   |   ├── evaluate_on_new_subjects_with_3_methods.py         # evaluates the 3 methods
|   |   ├── extract_keypoints_from_video.py                    # extracts keypoints from videos
|   |   ├── plot_three_evaluations.py                          # plots comparison graphs
│   │   └── score_eval_subjects.py                             # score the extracted evaluation videos against the fixed targets
│   │
│   ├── realtime_comparison/               # Live webcam inference
│   │   ├── realtime_comparison_squat.py
│   │   ├── realtime_comparison_lunge_movements.py
│   │   ├── realtime_comparison_limb_extensions.py
│   │   ├── realtime_comparison_jumping_jacks.py
│   │   └── realtime_inference_action_quality.py   # GRU/TCN-based variant
│   │
│   └── neural_network/                    # Quality-model training pipeline
│       ├── keypoint_normalize.py          # normalization (hip centering, shoulder scale)
│       │                                  # + mirroring shared by training and live inference
│       ├── generate_quality_labels.py     # step 1: quality labels per subject/action
│       ├── build_windowed_dataset.py      # step 2: sliding-window dataset (.npz)
│       ├── train_action_quality_net.py    # step 3a: GRU training
│       └── train_action_quality_tcn.py    # step 3b: TCN training
│
├── references/  (generated, git-ignored)  # Reference curves (.npy) for each exercise
├── GRU/ , TCN/  (generated, git-ignored)  # .npz datasets and model checkpoints
├── dataset/MMFi_Dataset/ (NOT included)   # MM-Fi dataset downloaded manually (see below)
├── yolov8n-pose.pt (NOT included)         # YOLOv8-Pose weights (auto-downloaded by Ultralytics)
├── LICENSE                                # Apache License 2.0
└── README.md

```

### Role of the Main Modules

- **`mmfi_lib/`**: wrapper for the MM-Fi dataset (file loading, PyTorch `Dataset`/`DataLoader`). It is not original project code: it is the official MM-Fi toolkit (see [References](#references)), reused to read the dataset's RGB 2D keypoint sequences.
- **`src/utils.py`**: the single place where geometry and scoring logic lives, so that offline evaluation on MM-Fi and live webcam evaluation use *exactly* the same logic.
- **`src/reference_extraction/`**: for each exercise, loads one or more MM-Fi sequences, computes the signal of interest (an angle, or a pair of signals), resamples it to a fixed length, and stores the resulting average reference curve in `references/`.
- **`src/evaluate_dataset/`**: evaluates *all* subjects in an MM-Fi environment against a target (fixed for the squat, empirical — i.e. estimated from the population mean — for the other exercises, since no published clinical target exists for them), and prints an accuracy report. It also helps identify "good" subjects to use as reference candidates.
- **`src/realtime_comparison/`**: captures webcam frames, runs YOLOv8-Pose frame by frame, tracks the active patient (`select_patient_keypoints`, robust to multiple people in the scene), calibrates thresholds from the patient's observed range of motion during the first few seconds, detects repetitions through a state machine (calibrating → standing/resting → moving/extending), and assigns a score to each repetition.
- **`src/neural_network/`**: a separate, alternative pipeline that does not rely on fixed joint-angle rules. Instead it trains a recurrent (GRU) or convolutional (TCN) network to predict both the exercise and a 0-100 quality score from normalized keypoint windows, using as a weak label the same heuristic score already computed by the rule-based pipeline.
- **`src/main.py`**: a Tkinter graphical interface that lets the user pick an exercise with one click and runs the corresponding reference-extraction and real-time comparison steps in sequence.

---

## Dataset

The project uses **MM-Fi**: *Multi-Modal Non-Intrusive 4D Human Dataset*, presented at NeurIPS 2023 in the Datasets and Benchmarks Track. It contains 40 subjects performing 27 daily and rehabilitation actions across five synchronized modalities: RGB, depth, LiDAR, mmWave, and WiFi-CSI.

This project uses **only the RGB modality**, specifically the annotated 2D keypoints, as the reference representation against which YOLOv8-Pose estimates from the webcam stream are compared.

- Project page: <https://ntu-aiot-lab.github.io/mm-fi>
- Official toolkit/repository (`mmfi_lib/` is derived from it): <https://github.com/ybhbingo/MMFi_dataset>
- Paper: Yang et al., *"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing"*, NeurIPS 2023, arXiv:2305.10345.

The dataset is **not included in this repository** because of its size and is excluded from Git through `.gitignore`. Download it using the links provided by the official MM-Fi repository and place it in the following location:

```
<project root>/dataset/MMFi_Dataset/
    E01/S01/A01/rgb/frame001.npy ...
    E01/S01/A01/ground_truth.npy
    ...
```
`src/utils.py` automatically defines `DATASET_ROOT = <PROJECT_ROOT>/dataset/MMFi_Dataset`.

---

## Evaluation Dataset

MM-Fi is used only for training, threshold calibration, and reference-curve construction; it is **not** used as the final evaluation set. A separate evaluation dataset was
therefore built from scratch, specifically for testing the three scoring
methods (rule-based, GRU, TCN) on data the models had never seen during
training or reference extraction.

### Composition

- **5 subjects** (`S1`-`S5`), none of which appear in the MM-Fi training
  or reference data.
- Each subject performed all **6 supported exercises**: squat,
  left/right lunge, left/right limb extension, jumping jacks.
- Each subject performed 3-5 repetitions per exercise.

### Recording conditions

- See See the [Camera and Scene setup](#camera-and-scene-setup) section below for the complete camera and ambient position position
- Videos were recorded at 30 FPS and processed with
  `extract_keypoints_from_video.py`, which re-samples frames to
  approximately the same ~10 fps keypoint rate used by MM-Fi
  (`MMFI_APPROX_FPS`), keeping the two data sources comparable.


The evaluation dataset itself is not included in the repository (see
`.gitignore`) for the same reason MM-Fi is not included.
It is possible to place an evaluation dataset of your own, with the same exercise taken into account, in the following location:
```
<project root>/eval_dataset/
    S1/jumping_jacks/confidence.npy
    S1/jumping_jacks/keypoints.npy 
    S1/limb_extension_left/confidence.npy
    ...
```
---
## Ground Truth

Physical therapy exercise correctness has no single objective
definition without a licensed physiotherapist's clinical judgment,
which is outside the scope of this project and this team's expertise.

The project does **not** attempt to define a clinically validated "correct"
movement. Instead, it fixes **one arbitrary but consistent reference
movement per exercise**, used identically across the rule-based,
GRU, and TCN pipelines so that the three methods remain fairly
comparable to each other -- not to a medical ground truth.

### How the reference movement is defined

For the squat, a single fixed clinical-looking value (95° of knee
flexion) is used as a simple, illustrative baseline target (see
`evaluate_dataset_fixed_targets_squat.py`).

For every other exercise (lunges, limb extensions, jumping jacks), no
such published reference exists, so the "correct" movement is instead
estimated **empirically from the MM-Fi population itself**: for each
signal of interest (e.g. minimum knee angle for a lunge, or the
95th-percentile leg-spread/arm-raise for jumping jacks), the target is
the **mean of that signal across all MM-Fi subjects** who performed the
exercise

This choice means:
- The "ground truth" quality label used to train and evaluate all three
  methods is a **population-relative heuristic score**
  (`calculate_depth_score()` in `utils.py`), not a clinically annotated
  label.
- A subject's score reflects how close their movement is to the
  *average* MM-Fi subject's movement for that exercise, not to a
  medically validated form.
---

## Installation

```bash
# 0. Create the virtual environment if it does not already exist
python -m venv venv

# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# If PowerShell reports an execution-policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then activate the virtual environment again

# 2. Install the dependencies
pip install ultralytics opencv-python scipy torch 
```

### Main Dependencies

- `torch` — training and inference of the GRU/TCN models.
- `ultralytics` — YOLOv8-Pose inference.
- `opencv-python` — webcam acquisition and video/image processing.
- `scipy` — reading `.mat` files used by parts of the MM-Fi toolkit; WiFi-CSI is not used by the active pipeline, but the dependency is required by `mmfi_lib`.
- `numpy` — numerical processing and storage of intermediate data.
- `pandas` — reading and manipulating the CSV comparison report in `plot_three_evaluations.py`.
- `matplotlib` — generating the comparison plots in `plot_three_evaluations.py`.
- `tkinter` — graphical user interface; included with standard Python installations on Windows.

The `yolov8n-pose.pt` weights are downloaded automatically by Ultralytics on first use. They can also be placed manually in the project root.

---

## How to Run the Project

### Option A — Graphical Interface (Recommended)

```bash
python src/main.py
```

A window opens with one button for each supported exercise. When an exercise is selected, the interface automatically runs the corresponding `reference_extraction_*.py` script when necessary and then launches the matching real-time comparison script from `realtime_comparison/`.

### Main Parameters

The following parameters can be modified at the beginning of the relevant scripts or in `src/utils.py`:

| Parameter | Location | Description |
|---|---|---|
| `SUBJECTS` | `reference_extraction_*.py` | MM-Fi subjects used to build the reference |
| `CAMERA_INDEX` | `realtime_comparison_*.py` | Webcam index |

### Option B — Offline Evaluation on the MM-Fi Dataset

```bash
python src/evaluate_dataset/evaluate_dataset_fixed_targets_squat.py
# One evaluation script is available for each exercise in src/evaluate_dataset/
```

The scripts calculate a score for each subject in environment `E01` and print a summary including the mean, standard deviation, and number of subjects within the selected tolerance.

### Option C — Neural Pipeline (GRU/TCN)

See the [Neural Pipeline](#neural-pipeline-gru-vs-tcn) section below for the complete execution order.
---
## Camera and Scene Setup

Since the system relies on a single RGB camera and 2D pose estimation (no depth or multi-view information), correct camera placement and body positioning are essential for reliable keypoint detection and, in turn, for accurate scoring. The guidelines below apply to all exercises unless stated otherwise.

### General Guidelines

- Distance from the camera: position yourself far enough that your whole body is visible in the frame, with some margin above your head and below your feet, at every point of the movement (including the deepest point of a squat or lunge, or arms fully raised in a jumping jack).
- Camera height: place the camera at roughly hip/torso height, on a stable support rather than handheld, to avoid a distorted viewing angle.
- Framing: keep the whole body inside the frame throughout the exercise. If a joint leaves the frame even briefly, that frame is discarded by the confidence check (keypoints_are_valid / KP_CONF_THRESHOLD) and cannot contribute to the score.
- Lighting: use even, front-facing lighting. Avoid strong backlighting (e.g. a bright window directly behind the subject), which makes the silhouette hard for YOLOv8-Pose to detect.
- Background: a plain, uncluttered background improves detection reliability, although it is not strictly required.
- Clothing: wear fitted clothing that contrasts with the background, so joints are clearly visible; loose or baggy clothing can hide the true position of elbows, knees, and shoulders.
- People in the scene: ideally, only the patient should be in frame. The system includes a tracking mechanism (select_patient_keypoints) to keep following the same person if someone else (e.g. a caregiver) briefly enters the scene, but a single, uncluttered subject gives the most reliable results.

(Reference images showing the ideal setup and camera framing for each exercise will be added here.)
---
## System Input and Output

### Input

- Local webcam stream using a configurable camera index, with the device's default resolution and frame rate.
- For offline processing, RGB sequences from MM-Fi (`.npy` files containing 17 COCO keypoints × `(x, y)`).

### Output

- Real-time video overlay showing the current state of the state machine (calibration, movement, or rest), the current repetition quality score from 0 to 100% (color-coded green/yellow/red based on threshold), and calibration/warning messages.
- A terminal log entry for each completed repetition, for example:

```text
Rep: depth_achieved=... score=...%
```

- For the neural pipeline: the predicted exercise and quality score from 0 to 100, updated approximately once per second during live inference.

### Explicit Scope

The system does **not** provide a clinical diagnosis, does not replace assessment by a physiotherapist, and does not guarantee correct recognition when multiple people are present in the scene beyond the tracking mechanism already implemented, which is designed to keep the closest/most central person as the active patient.

---

## MM-Fi Dataset Evaluation Pipeline

For each exercise, the evaluation workflow is:

1. **`evaluate_dataset_fixed_targets_*.py`** — loads the subjects from an MM-Fi environment, computes the relevant signal or signals, estimates an exercise target, and calculates a subject-level score using the same `calculate_depth_score()` function used during live inference. The squat uses a fixed target of 95°, while the other exercises use empirical population-based targets because no published clinical reference is available. The scripts also identify subjects within the selected tolerance who can be considered as reference candidates.
2. **`reference_extraction_*.py`** — takes one or more selected MM-Fi subjects (ideally the "good" ones from step 1), resamples their sequences to a fixed length of 100 points, and averages them into a single reference curve stored in `references/`.
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
python src/realtime_comparison/realtime_inference_action_quality.py   # live inference
```

### Pipeline Steps

- **`generate_quality_labels.py`** recalculates the same heuristic quality score for *all 40* subjects across environments E01–E04 (not just E01, unlike the `evaluate_dataset/` scripts). It does not introduce a new definition of correct exercise execution.
- **`build_windowed_dataset.py`** normalizes each sequence by centering it on the mid-hip and scaling it according to shoulder width, as implemented in `keypoint_normalize.py`. The sequences are divided into sliding windows of 30 frames with a stride of 10. Left/right mirroring is also applied as data augmentation, so a mirrored left-lunge sample can be used as a synthetic right-lunge sample. The train/validation split is performed **by subject rather than by window**, ensuring that validation includes subjects not seen during training.
- **`train_action_quality_net.py`** and **`train_action_quality_tcn.py`** use the same overall architecture: a shared encoder followed by two output heads, one for exercise classification and one for quality-score regression. The only architectural difference is the encoder: recurrent GRU versus convolutional TCN, so the comparison is fair.

### Important Training Limitation

The neural network does **not** learn a clinical definition of correct exercise form. Instead, it learns to reproduce the same population-relative heuristic score already used by the rule-based pipeline.

The quality target is therefore a **weak label**, not a clinically validated ground-truth score.

---

## Metrics and Baselines

- **Quality score metric**: the two-zone `calculate_depth_score()` function assigns scores from 100% to 80% linearly within `DEPTH_TOLERANCE`, then decreases from 80% to 0% across `DEPTH_FALLOFF_RANGE`. This is an internal project metric used consistently in both offline and online evaluation.
- **Baseline**: the **rule-based/geometric method** serves as the
  project's baseline condition: the squat uses a fixed target of 95° as a simple baseline based on the clinical value used by the project. For the other exercises, where no published clinical target is available, the baseline is the empirical target calculated as the mean of the MM-Fi population by `evaluate_dataset_fixed_targets_*.py`.
- **Model comparison**: three distinct approaches are compared on the same subjects/actions — the rule-based/geometric method (a traditional CV approach) and the two deep-learning models, GRU and TCN — using classification accuracy, quality-score MAE on the 0–100 scale, parameter count, and CPU latency per window. This comparison is carried out by the script 'evaluate_on_new_subjects_with_3_methods', using the reference curves in `references/` (rule-based) and the two checkpoints produced by `train_action_quality_net.py` / `train_action_quality_tcn.py` (GRU/TCN) as inputs. The resulting numbers and discussion are reported in the final presentation/project report, not in this repository.
- **Pose-estimation metrics**: `mmfi_lib/evaluate.py` also provides the standard MPJPE and PA-MPJPE metrics from the original MM-Fi toolkit, including Procrustes alignment, for a possible direct evaluation of pose-estimation quality against the dataset's 3D ground truth.

---

## Third-Party Code and Libraries

| Component | Source | License | Role in the Project |
|---|---|---|---|
| `mmfi_lib/mmfi.py`, `mmfi_lib/evaluate.py` | Official MM-Fi toolkit (<https://github.com/ybhbingo/MMFi_dataset>) | See original repository | Dataset loading, PyTorch `Dataset`/`DataLoader`, MPJPE/PA-MPJPE metrics |
| YOLOv8-Pose (`yolov8n-pose.pt`) | Ultralytics (<https://github.com/ultralytics/ultralytics>) | AGPL-3.0 | Real-time 2D keypoint estimation from the webcam stream |
| PyTorch | <https://pytorch.org/> | BSD-style | GRU/TCN training and inference |
| OpenCV (`opencv-python`) | <https://opencv.org/> | Apache 2.0 | Video acquisition and visualization, image/depth processing |
| SciPy | <https://scipy.org/> | BSD | Reading `.mat` files used by the MM-Fi toolkit |
| This project | — | Apache License 2.0 | See `LICENSE` |

---

## AI/LLM Tool Usage

AI tools were used throughout this project in the following ways:

- Code generation and programming assistance: Claude (Anthropic) was used extensively for writing and structuring the Python code in this repository. As the team is not highly experienced with Python, Claude was relied upon to translate the project's logic (geometric scoring, dataset handling, model training pipelines, real-time webcam scripts, etc.) into working code.
- Debugging: AI assistance was used to understand and resolve issues that came up while developing and running the code, from interpreting error messages to identifying the root cause of unexpected behavior.
- Image generation: ChatGPT (OpenAI) was used to generate images.
- Documentation: this README.md file was prepared with the assistance of Claude (Anthropic), based on the project's source code and the course guidelines provided by the instructors; the technical content reflects the implementation actually present in the repository at the time of writing.

All AI-generated code was reviewed, tested, modified and validated by the authors before being included in the final submission.

---

## References

- J. Yang, H. Huang, Y. Zhou, X. Chen, Y. Xu, S. Yuan, H. Zou, C. X. Lu, L. Xie. *"MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing"*. NeurIPS 2023, Datasets and Benchmarks Track. [arXiv:2305.10345](https://arxiv.org/abs/2305.10345) · [Project page](https://ntu-aiot-lab.github.io/mm-fi) · [Repository/toolkit](https://github.com/ybhbingo/MMFi_dataset)
- Ultralytics YOLOv8 (pose estimation). [Documentation](https://docs.ultralytics.com/tasks/pose/) · [Repository](https://github.com/ultralytics/ultralytics)
- Procrustes alignment / PA-MPJPE: implementation adapted in `mmfi_lib/evaluate.py` from a common MATLAB `procrustes` implementation ported to NumPy, following a standard approach in 3D human pose estimation.

---

## Known Limitations

- The targets used for the exrcises are **empirical estimates** based on the mean of the MM-Fi population. They are not clinically validated targets.
- The **GRU/TCN neural network does not learn a clinical definition of correct form**. It learns to reproduce the same heuristic score generated by the rule-based pipeline.
