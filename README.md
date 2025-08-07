# Task 7: Video Object Tracking

## Objective

* Load and process video data.
* Apply multi-object detection on video frames.
* Track multiple objects across frames using tracking algorithms.
* Visualize and export results as annotated videos.
* Document the entire pipeline in a structured and reproducible Jupyter notebook.

## Data Selection
<img width="1007" height="172" alt="image" src="https://github.com/user-attachments/assets/6740f4e0-8a2b-486b-ac92-6588a9a55c1e" />


We selected three video sequences to cover a variety of tracking scenarios:

| Sequence              | Duration (s) | FPS  | Frames | Resolution  | Classes | Unique IDs |
|-----------------------|-------------:|-----:|-------:|-------------|---------|-----------:|
| people-tracking [1]   |         1.37 | 30.0 |     41 | 1280×720    | person  |         45 |
| MOT17-09-FRCNN  [2]   |        17.50 | 30.0 |    525 | 1920×1080   | person  |         64 |
| MOT17-13-FRCNN  [3]   |        30.00 | 25.0 |    750 | 1920×1080   | person  |        188 |

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


## References 
[1]: https://www.kaggle.com/datasets/trainingdatapro/people-tracking
[2]: https://motchallenge.net/vis/MOT17-09-FRCNN
[2]: https://motchallenge.net/vis/MOT17-13-FRCNN
