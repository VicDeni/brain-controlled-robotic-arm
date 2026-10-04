# Brain-Controlled Robotic Arm: EEG-Based Movement Intent Decoding with Kalman-Filtered Robotic Control

An end-to-end pipeline that decodes a person's imagined movement directly from their brain activity (EEG) and uses it to control a simulated robotic arm — the same kind of signal chain underlying real assistive technologies like brain-controlled wheelchairs and prosthetics.

## Overview

A Brain-Computer Interface (BCI) lets a person control a device using brain activity alone, with no physical movement required. This project builds one from the ground up: real EEG data is recorded while a person imagines moving their left hand, right hand, or feet; a machine learning model learns to decode which movement is being imagined from the raw brain signal; and a Kalman filter smooths the model's noisy, real-time predictions into a stable control signal that drives a simulated robotic arm.

The project is built around real, publicly available EEG data rather than synthetic signals, which introduces a genuinely harder and more realistic problem than a clean simulation: brain signals are noisy, vary from person to person, and never produce perfectly separable classes. Working with that messiness honestly, rather than chasing an unrealistically clean result, is part of the point.

## Project Structure

### Phase 1 — EEG Signal Processing & Exploration
`notebooks/01_eeg_preprocessing_exploration.ipynb` *(planned)*

- Loads real motor-imagery EEG recordings from the PhysioNet EEG Motor Movement/Imagery Dataset via the MNE-Python library.
- Explains what an EEG signal actually is and how it's recorded.
- Applies filtering (bandpass/notch) to clean the raw signal and remove noise.
- Segments the continuous recording into labeled trials (left hand, right hand, feet, rest) and visualizes what imagined-movement brain activity looks like.

### Phase 2 — Movement Intent Classification
`notebooks/02_movement_classification.ipynb` *(planned)*

- Extracts features from the cleaned EEG signal (e.g. band power, spatial patterns).
- Trains a classifier to decode which movement a person is imagining from their brain activity alone.
- Evaluates real-world classification accuracy on held-out data, with an honest discussion of what realistic BCI performance looks like.

### Phase 3 — Kalman Filter Smoothing
`notebooks/03_kalman_filter_smoothing.ipynb` *(planned)*

- Treats the classifier's raw, noisy, trial-by-trial predictions as a noisy signal in their own right.
- Applies a Kalman filter to smooth this prediction stream into a stable estimate of intended movement over time, reducing erratic, single-trial misclassifications from translating directly into erratic robot movement.

### Phase 4 — Simulated Robotic Arm Control
`notebooks/04_robotic_arm_control.ipynb` *(planned)*

- Builds a simple simulated robotic arm in Python.
- Maps the Kalman-filtered movement intent into actual arm movement commands, closing the loop from raw brain signal to robotic motion.
- Visualizes the full pipeline running end-to-end: EEG in, robot arm movement out.

## Key Concepts Demonstrated

- EEG signal processing: filtering, epoching, and artifact handling
- Brain-computer interface (BCI) motor-imagery classification
- Kalman filtering for real-time noise reduction and state estimation
- Simulated robotic kinematics and control
- Working with real, noisy, human-subject data rather than synthetic/simulated signals

## Data

[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — accessed directly through MNE-Python's built-in dataset loader, no manual download required.

## Setup

```bash
git clone https://github.com/VicDeni/brain-controlled-robotic-arm.git
cd brain-controlled-robotic-arm
python3 -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then open any notebook in VS Code or Jupyter and run the cells from top to bottom.

## Status

- 🚧 Phase 1 planned
- 🚧 Phase 2 planned
- 🚧 Phase 3 planned
- 🚧 Phase 4 planned
