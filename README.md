# ChronoTrace AI

**Multi-Camera Forensic Timeline Reconstruction**

ChronoTrace AI is a Python-based prototype that helps investigators compare videos from multiple cameras and reconstruct a shared event timeline when camera clocks and frame rates differ.

## The Problem

Traffic incidents, factory accidents, and security breaches may be recorded by several CCTV cameras, smartphones, dashcams, and traffic cameras. Differences in camera clocks, frame rates, and recording conditions make manual synchronization difficult.

## Our Solution

ChronoTrace AI provides a workflow for video metadata extraction, visual event analysis, candidate cross-camera matching, clock-offset estimation, and timeline generation.

## Features

- Video metadata extraction using FFmpeg and OpenCV.
- Frame extraction and scene-change detection.
- Perceptual image hashes and visual similarity features.
- Clock-offset and clock-rate estimation using a linear time model.
- Robust outlier handling with RANSAC.
- Multi-camera timeline visualization using Streamlit.
- SHA-256 evidence-file integrity checks and audit logging.
- CSV and JSON report exports.
- Automated tests, including controlled synthetic scenarios.

## Technology Stack

- Python 3.11+
- OpenCV for frame extraction and image processing
- FFmpeg / imageio-ffmpeg for video metadata and container inspection
- NumPy and SciPy for time-offset estimation and robust linear regression
- PySceneDetect for scene-change detection
- Streamlit for interactive timeline dashboard
- Pandas for structured forensic exports
- Pytest for test suite validation

## Project Structure

```text
Hit and run case/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── evidence_integrity.py  # SHA-256 custody, immutable vault, audit_log.jsonl
│   ├── video_metadata.py      # FFmpeg (bundled) & OpenCV container/codec extraction
│   ├── frame_extractor.py     # PTS timestamp frame extraction & thumbnail caching
│   ├── event_detector.py      # PySceneDetect scene cuts & visual spike detector
│   ├── visual_matcher.py      # dHash, pHash (DCT), HSV color histograms & candidates
│   ├── time_aligner.py        # RANSAC linear drift model (T_ref = offset + rate * t_src)
│   ├── timeline_builder.py    # Multi-camera timeline aggregation & query playhead
│   ├── report_exporter.py     # CSV & JSON forensic export package generator
│   └── synthetic_generator.py # 3-camera controlled hit-and-run dataset generator
├── tests/
│   ├── test_evidence_integrity.py
│   ├── test_video_metadata.py
│   ├── test_visual_matcher.py
│   ├── test_time_aligner.py
│   ├── test_exports.py
│   └── test_synthetic_end_to_end.py
├── sample_data/               # Pre-generated 3-camera synthetic video fixtures
└── reports/                   # Destination for exported forensic data packages
```

## Installation

Use Python 3.11 or another version supported by the pinned dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

## Run Tests

```powershell
python -m pytest -v
```

All 23 automated unit and end-to-end tests validate:
- Cryptographic SHA-256 hashing and tamper detection.
- Handling of corrupt, 0-byte, and non-video files.
- Perceptual hash and visual similarity invariance.
- RANSAC rejection of false visual match outliers.
- Clock offset and clock-rate drift estimation under different frame rates (15, 24, 30 FPS).
- Empirical measurement of timing error against ground truth (Median error: 0.00000s, 95th-percentile error: 0.00000s).

## Demo

Use the application's synthetic demo dataset to explore multi-camera alignment, visual match candidates, and timeline navigation. In the Streamlit sidebar, click **"🚀 Load Synthetic Demo Case (3 Cams)"** to load pre-configured footage representing a multi-camera traffic collision scene.

Synthetic results are not evidence of real-world forensic accuracy.

## Reports

The application generates four standard forensic artifacts in the `reports/` directory:
- `unified_timeline.csv`: Master timeline with event categories, source and corrected timestamps, frame indices, and review statuses.
- `event_matches.csv`: Detailed log of cross-camera visual candidate matches, Hamming distances, inlier/outlier statuses, and forensic disclaimers.
- `synchronization_report.json`: Mathematical calibration formulas, RANSAC inlier/outlier counts, drift rates in PPM, residual RMSEs, and limitations disclosures.
- `evidence_manifest.json`: Cryptographic SHA-256 evidence catalog, file sizes, stream parameters, and audit trail.

## Limitations

- Visual similarity does not establish that two frames depict the same event.
- Reliable synchronization depends on trustworthy reference timestamps or sufficient verified matching events.
- Clock drift and timing uncertainty must be estimated from available evidence.
- Real-world performance depends on camera quality, lighting, occlusion, dropped frames, and scene overlap.
- This prototype does not establish legal admissibility or replace a qualified forensic examiner.

## Evidence and Privacy

Use synthetic or authorized footage for testing. Do not publish private video evidence, personal information, credentials, or confidential investigation records.

Preserve original evidence files and validate their integrity independently. Hashing alone does not prove the complete chain of custody.

## Project Status

Prototype for experimentation and demonstration. See the test results and known limitations before using it on real investigations.
