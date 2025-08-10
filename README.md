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
<img width="1862" height="1322" alt="image" src="https://github.com/user-attachments/assets/7721b684-c70d-4a1b-9ff7-dc15421600ff" />
<br/>

• only image files are loaded (`.jpg/.jpeg/.png`) 
• Random sample 4 frames 
• BGR to RGB fix before plotting (avoid weird colors in matplotlib)  
• resized frames only (visual check) so no numeric normalization here  
• consistent frame size from `img1_resized/` helps stable detection/tracking   
• simple switch `seq = SEQUENCES[0]` to rotate datasets fast while keeping the same pipeline


## Model Building
<img width="1942" height="1325" alt="image" src="https://github.com/user-attachments/assets/e546579e-b8a5-42bd-8322-1399271e7cff" />
<br/>
This run applies a YOLOv8 detector with ByteTrack to the MOT17-09-FRCNN sequence:

- Input: standardized frames from `img1_resized/` to keep resolution consistent across the sequence; tracking restricted to the person class.
- Detector: `yolov8l.pt` is selected for a balanced accuracy–speed trade-off; `yolov8x.pt` can be used when higher recall is required and compute allows.
- Tracker: ByteTrack (via `bytetrack.yaml`) handles data association to maintain stable IDs in crowded scenes.
- Key parameters: `conf = 0.3` favors recall on small/occluded pedestrians; `iou = 0.6` relaxes NMS to retain nearby boxes; `classes = [0]` limits processing to persons. Optionally use `imgsz = 1280` for finer detections.
- Outputs: an annotated video and MOT-format `labels/*.txt` files are written under `.../bytetrack/run/`, ready for downstream evaluation (e.g., MOTA/MOTP).
- Tuning guidance: raise `conf` to increase precision, tighten `iou` to reduce overlapping boxes, expand `classes` for multi-class tracking, and adjust ByteTrack thresholds if ID switches are frequent.

This configuration prioritizes high recall and ID stability while producing clean, reproducible artifacts for evaluation and reporting.

<br/>
<br/>

<img width="1932" height="1086" alt="image" src="https://github.com/user-attachments/assets/2248c192-64cc-4cbb-b269-a4c07c39ab96" />
<br/>

This run extends detection and tracking to **two classes** (persons and cars) on the MOT17-13-FRCNN sequence:

- Input: standardized frames from `img1_resized/` to keep resolution consistent across the sequence.
- Detector: `yolov8l.pt` for a balanced accuracy–speed profile; upgrade to `yolov8x.pt` if higher recall is needed and compute allows.
- Tracker: ByteTrack (`bytetrack.yaml`) performs data association to maintain stable IDs across frames for **multi-class** targets.
- Key parameters:
  - `classes = [0, 2]` → track **person (0)** and **car (2)** only; reduces noise from irrelevant categories.
  - `conf = 0.05` → very low threshold to capture small/distant cars and partially occluded pedestrians; expect more false positives.
  - `iou = 0.6` → relaxed NMS to retain nearby boxes in crowded traffic; helps reduce missed detections but may increase overlaps.
  - (Optional) `imgsz = 1280` for finer detections if VRAM permits.
- Outputs: annotated video and MOT-format `labels/*.txt` (per-frame tracks) are written to `.../bytetrack_cars_and_ppl/run/`, ready for downstream metrics or visualization.
- Tuning guidance:
  - Increase `conf` (e.g., 0.2–0.3) to improve precision if false positives are high.
  - Tighten `iou` (e.g., 0.5) to suppress overlapping boxes in low-density scenes.
  - Expand or change `classes` if additional categories (e.g., bicycle, motorcycle) are needed for your analysis.

This configuration prioritizes **recall** for mixed pedestrian-vehicle scenes, enabling more complete trajectory coverage before later precision filtering or post-processing.
<br/>
<br/>
<img width="1949" height="1034" alt="image" src="https://github.com/user-attachments/assets/8b6df37a-607c-4407-b057-6c7c3f8e1dfe" />
<br/>

This run applies a YOLOv8 detector with ByteTrack on the people-tracking sequence:

- Input: preprocessed, resized frames from `data/processed/resized_images` to ensure uniform resolution.
- Detector: `yolov8x.pt` with `imgsz = 1280` to improve small/occluded person recall; accepts the runtime/VRAM cost for accuracy.
- Tracker: ByteTrack (`bytetrack.yaml`) for robust ID assignment across frames.
- Key parameters:  
  - `conf = 0.3` to suppress low-confidence noise while retaining moderate detections.  
  - `iou = 0.5` to tighten NMS and reduce overlapping boxes.  
  - `classes = [0]` to restrict processing to the person class only.
- Outputs: annotated video and MOT-format `labels/*.txt` written to `.../bytetrack_highacc/run_highacc/`, suitable for downstream metrics (e.g., MOTA/MOTP) and qualitative review.
- Tuning guidance: raise `conf` for higher precision; relax `iou` (e.g., 0.6) if fragmentation occurs in crowded scenes; if speed is a concern, downscale `imgsz` or switch to `yolov8l.pt`.


## Evaluation & Visualization
<img width="715" height="349" alt="image" src="https://github.com/user-attachments/assets/57d27874-149a-48bf-858a-a88bb4e369ec" />
<br/>
<br/>
**MOT17-09-FRCNN**
- MOTA ≈ 0 (slightly negative) + IDF1 = 0 → no effective matches to ground truth.
- MOTP = NaN → zero true-positive matches (precision undefined).
- 0 switches likely reflects “no matched tracks” rather than stable IDs.
- Very high misses → recall failure or evaluation mismatch.
<br/>
<video controls src="output/samples/mot17-09_demo.mp4" width="720"></video>
<p><a href="output/samples/mot17-09_demo.mp4">download / watch the tracked video</a></p>
<br/>
<br/>
<img width="575" height="297" alt="image" src="https://github.com/user-attachments/assets/347aea64-ac00-4202-9571-e463fa1851ba" />
<br/>
<br/>
**MOT17-13-FRCNN**
- MOTA ≈ 0 (slightly negative) + IDF1 = 0 → no true matches to GT.
- MOTP = NaN → no true positives (precision undefined).
- 0 switches likely because no matches were established.
- Very high misses → recall failure or eval mismatch.
<br/>
<video controls src="output/samples/cars_and_ppl_mot17-13_demo.mp4" width="720"></video>
<p><a href="output/samplescars_and_ppl_mot17-13_demo.mp4">download / watch the tracked video (car and person)</a></p>

<video controls src="output/samples/mot17-13_demo.mp4" width="720"></video>
<p><a href="output/samples/mot17-13_demo.mp4">download / watch the tracked video (person only)</a></p>
<br/>
<br/>
<img width="1020" height="288" alt="image" src="https://github.com/user-attachments/assets/73642703-4ac5-44f6-b1ca-3072d002d680" />
<br/>
**people-tracking**

• 41 frames; gt (1785×6), hyp (1585×6) → partial but close  
• MOTA ≈ 0.139 so baseline; idf1 ≈ 0.261 → some id consistency  
• MOTP = 0.088 so check box scale/units and conversion pipeline  
• confirm frame indexing (1-based vs 0-based) and person-only filtering  
<br/>
<video controls src="output/samples/tracked_ppl.mp4" width="720"></video>
<p><a href="output/samples/tracked_ppl.mp4">download / watch the tracked video</a></p>

## Project Structure

```
Task7/
├── data/
│ └── raw/
│ ├── MOT17-09-FRCNN/
│ ├── MOT17-13-FRCNN/
│ ├── people-tracking/
├── notebooks/
│ └── video_tracking_action.ipynb
├── output/
│ ├── samples/
│ │ ├── mot17-09_demo.mp4
│ │ ├── mot17-13_demo.mp4
│ │ ├── cars_and_ppl_mot17-13_demo.mp4
│ │ └── tracked_ppl.mp4
│ └── output.zip # full outputs (compressed)
├── .gitattributes # For large files
└── README.md
```

## Requirements

* Python 3.8 or higher
* OpenCV
* PyTorch
* Ultralytics YOLOv8
* NumPy
* Pandas
* Matplotlib


## Usage

1. Place raw videos in the `data/raw/` directory.
2. Launch the pipeline notebook:

   ```bash
   jupyter notebook notebooks/video_tracking_action.ipynb
   ```
3. Run all cells to execute detection, tracking, and visualization steps.
4. View annotated videos in the `output/` folder.


## References 
Datasets used:

https://www.kaggle.com/datasets/trainingdatapro/people-tracking [1]
https://motchallenge.net/vis/MOT17-09-FRCNN [2]
https://motchallenge.net/vis/MOT17-13-FRCNN [3]
