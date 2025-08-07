# Task 7: Video Object Tracking

## Objective

* Load and process video data.
* Apply multi-object detection on video frames.
* Track multiple objects across frames using tracking algorithms.
* Visualize and export results as annotated videos.
* Document the entire pipeline in a structured and reproducible Jupyter notebook.

## Data Selection

* **Source Type:** Sample videos with human or object movement (e.g., sports, surveillance).
* **Video Duration & Resolution:** Record duration (in seconds or minutes) and resolution (e.g., 1920×1080).
* **Frame Rate:** Note frames per second (fps) for each video.
* **Objects & Actions:** Specify the number and type of objects/actions involved.
* Include at least one video containing multiple moving objects (e.g., cars and people) to evaluate multi-object tracking performance.

## Data Preprocessing

* Extract frames from raw videos.
* Resize and normalize frames as needed.
* Visualize a sample of frames to verify correctness.

## Model Building

* **Object Detection:** Use pre-trained models such as YOLO, SSD, or Faster R-CNN.
* **Tracking Algorithms:** Apply SORT, Deep SORT, or ByteTrack for multi-object tracking.
* Save annotated videos demonstrating tracked bounding boxes.

## Evaluation & Visualization

* Compute tracking metrics like MOTA (Multiple Object Tracking Accuracy) and MOTP (Multiple Object Tracking Precision).
* Compare original video frames versus tracked outputs side-by-side.
* Present evaluation results on at least one multi-object video to demonstrate handling of complex scenarios.

## Report

* A finalized Jupyter notebook (`notebooks/video_tracking_action.ipynb`) containing the full detection, tracking, action analysis, and visualization pipeline.
* Annotated video samples stored in `output/tracked_videos/`.
* Figures illustrating detection examples, tracking results, and action summaries.
* A concise summary of methodology and results.

## Project Structure

```
project-root/
├── data/
│   └── raw/                        # Raw input videos
├── notebooks/
│   └── video_tracking_action.ipynb  # Full pipeline notebook
├── output/
│   ├── tracked_videos/             # Annotated video outputs
│   └── action_visualizations/      # Visual summaries of actions
├── figures/
│   ├── detection_examples.png
│   ├── tracking_results.png
│   └── action_summary.png
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Requirements

* Python 3.8 or higher
* OpenCV
* PyTorch
* Ultralytics YOLOv8
* NumPy
* Pandas
* Matplotlib

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project-root

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your raw videos in the `data/raw/` directory.
2. Launch the pipeline notebook:

   ```bash
   jupyter notebook notebooks/video_tracking_action.ipynb
   ```
3. Run all cells to execute detection, tracking, and visualization steps.
4. View annotated videos and figures in the `output/` and `figures/` folders.
