# Brain-Controlled Robotic Arm: EEG-Based Movement Intent Decoding with Kalman-Filtered Robotic Control

An end-to-end pipeline that decodes a person's imagined movement directly from their brain activity (EEG) and uses it to control a simulated robotic arm — the same kind of signal chain underlying real assistive technologies like brain-controlled wheelchairs and prosthetics.

## Overview

A Brain-Computer Interface (BCI) lets a person control a device using brain activity alone, with no physical movement required. This project builds one from the ground up: real EEG data is recorded while a person imagines moving their left or right fist; a machine learning model learns to decode which movement is being imagined from the raw brain signal; and a Kalman filter smooths the model's noisy, real-time predictions into a stable control signal that drives a simulated robotic arm sorting objects into bins.

The project is built around real, publicly available EEG data rather than synthetic signals, which introduces a genuinely harder and more realistic problem than a clean simulation: brain signals are noisy, vary from person to person, and never produce perfectly separable classes. Working with that messiness honestly, rather than chasing an unrealistically clean result, is part of the point.

![Pipeline](images/bci_pipeline.png)

## Project Structure

### Phase 1 — EEG Signal Processing & Exploration ✅
`notebooks/01_eeg_preprocessing_exploration.ipynb`

- Loads real motor-imagery EEG (PhysioNet EEG Motor Movement/Imagery Dataset) via MNE-Python.
- Applies an 8–30 Hz bandpass filter and cuts the recording into labeled left-fist / right-fist trials.
- Shows event-related desynchronization (ERD): the mu/beta rhythm weakens over the motor cortex during imagined movement.

### Phase 2 — Movement Intent Classification ✅
`notebooks/02_movement_classification.ipynb`

- Extracts features with Common Spatial Patterns (CSP) and classifies with Linear Discriminant Analysis.
- Evaluates honestly with repeated cross-validation, with CSP fit inside each fold, plus a permutation test.
- Results: **71.8%** for subject 1 (p = 0.03, chance ≈ 51%); **59.7% ± 19.1%** averaged over subjects 1–10, ranging from chance to 98% depending on the person.
- Tests whether covariance shrinkage helps (it doesn't) and reports that honestly.

### Phase 3 — Kalman Filter Smoothing ✅
`notebooks/03_kalman_filter_smoothing.ipynb`

- Turns the decoder into a real-time stream: a sliding 1.5 s window, one guess every 0.1 s, evaluated out-of-sample by splitting on trials.
- Applies a 1-D Kalman filter (the same predict/update structure as the orbital project) to the stream of guesses.
- Across five subjects it cuts output jitter by 19–57% and mid-trial slips by 22–46% with almost no added delay, without changing accuracy. Smoothing steadies a decoder but can't make a chance-level one correct.
- Reports the smoothness vs. speed trade-off by sweeping the process noise Q.

![Kalman results](images/kalman_subject_comparison.png)

### Phase 4 — Brain-Controlled Sorting Task ✅
`notebooks/04_robotic_arm_control.ipynb`

The filtered intent signal drives a simulated 3-joint robotic arm that **sorts objects into a left or right bin**. The person imagines a left or right fist, and the hand slides toward the matching bin like a joystick: the decoded intent sets the hand's *speed*, and a small deadzone means weak evidence does nothing. When the hand reaches a bin, that counts as a selection.

![Brain-controlled sorting demo](images/sorting_demo.gif)

*Same brain signal, two arms: raw decoder (left) vs. Kalman-filtered (right). First 8 trials of subject 2, shown in order, not hand-picked.*

- **Closed loop:** EEG → decoder → Kalman filter → hand velocity → inverse kinematics → arm.

  ![Control loop](images/arm_control_loop.png)
- **Control:** hand velocity = g · deadzone(x̂), with gain g = 1.5 and deadzone d = 0.3, both fixed in advance rather than tuned on results. The hand and the filter reset at every cue.
- **Inverse kinematics:** for a hand target, the base yaw, shoulder and elbow angles are solved in closed form (law of cosines, elbow-up), so the arm follows the hand smoothly.

  ![Inverse kinematics geometry](images/ik_geometry.png)
- **Task metrics:** correct / wrong / no selection, time to select, information transfer rate (Wolpaw ITR), and *wobble*, which measures how much the commanded speed jitters beyond its net change.

**Results: five subjects, 45 trials each (decoded out-of-fold)**

| Arm | Correct | Wrong | No selection | Time to select | ITR (bits/min) | Wobble |
|---|---|---|---|---|---|---|
| Raw decoder | 56.0% | 18.2% | 25.8% | 2.93 s | 4.4 | 1.248 |
| Kalman (never reset) | 48.9% | 12.9% | 38.2% | 3.09 s | 3.5 | 0.492 |
| **Kalman (reset at cue)** | 51.6% | 12.4% | 36.0% | 3.02 s | 4.4 | 0.548 |

| Subject | Correct (raw) | Correct (Kalman) | Wobble (raw) | Wobble (Kalman) |
|---|---|---|---|---|
| 7 | 95.6% | 95.6% | 0.223 | 0.137 |
| 2 | 80.0% | 80.0% | 0.738 | 0.406 |
| 1 | 31.1% | 28.9% | 1.442 | 0.644 |
| 3 | 31.1% | 24.4% | 1.442 | 0.559 |
| 5 | 42.2% | 28.9% | 2.397 | 0.993 |

![Subject benchmark](images/arm_subject_benchmark.png)

**Takeaways**

- The Kalman filter makes the arm about **56% steadier** and cuts wrong selections from 18.2% to 12.4%. The arm holds still when the evidence is shaky instead of committing.
- The price is fewer correct selections (56.0% → 51.6%) and about 0.1 s more time per selection. With reset at each cue, ITR matches the raw decoder.
- The arm works well for subjects with a decodable signal (subjects 7 and 2) and **not at all for subjects 1, 3 and 5**, whose decoders are near chance. Smoothing cannot create information that isn't in the signal.

## Limitations

- Everything is simulated and replayed offline from recorded EEG. There is no live closed loop and no physical arm.
- The Kalman noise term R is calibrated using the true labels, and the process noise Q was tuned on subject 7 only.
- ITR ignores the rest periods between trials, so it is an upper bound.
- Decoder quality varies enormously between people: the pipeline works for some subjects and fails for others.

## Key Concepts Demonstrated

- EEG signal processing: filtering, epoching, and artifact handling
- Brain-computer interface (BCI) motor-imagery classification
- Kalman filtering for real-time noise reduction and state estimation
- Inverse kinematics, closed-loop control, and information transfer rate (ITR) for a simulated arm
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

- ✅ Phase 1 complete
- ✅ Phase 2 complete
- ✅ Phase 3 complete
- ✅ Phase 4 complete