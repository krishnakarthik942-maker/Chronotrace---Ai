# ChronoTrace AI — Technical Forensic Audit & Architecture Review

**Date of Audit:** October 26, 2026
**Auditor Role:** Senior Python Engineer, Computer-Vision Engineer, and Digital-Forensics Software Reviewer
**Target Repository:** ChronoTrace AI — Multi-Camera Forensic Timeline Reconstruction
**Environment Inspected:** Windows 11, Python 3.11.9, Pytest 9.1.1, OpenCV 5.0.0.93 / opencv-python-headless, Streamlit 1.66.1
**Audit Scope:** Full system architecture, evidence custody, cryptographic hashing, timestamp parsing, event detection, visual matching, clock synchronization algorithms (RANSAC / linear regression), multi-track timeline construction, report exports, unit & integration test coverage, and deployment safety.

---

## Executive Summary

ChronoTrace AI contains a solid architectural foundation for multi-camera video timeline alignment, featuring chunked SHA-256 cryptographic hashing, PySceneDetect integration, discrete cosine transform (DCT) perceptual hashing (pHash/dHash), color histogram matching, and a RANSAC-based linear clock drift model ($T_{\text{ref}} = \text{offset} + \text{rate} \times t_{\text{src}}$).

However, a rigorous digital-forensics and software-engineering review identified **critical evidentiary flaws, unhandled exceptions, resource leaks, git custody leakage vulnerabilities, and untested subsystem components** that must be resolved before this tool can be considered forensically reliable or safe for production deployment.

Most critically:
1. **Anchor Transitivity & Reference Assumption Defect:** When investigators confirm cross-camera candidate matches between two non-reference cameras (e.g., Camera 2 and Camera 3), the application treats Camera 2's uncalibrated local timestamp as the absolute reference timeline time, corrupting downstream timeline alignment.
2. **Fabricated Review Status in Exported Events:** In `app.py`, unreviewed candidate matches are passed into the timeline builder as confirmed matches, causing unreviewed events to be falsely marked with `review_status = "CONFIRMED"` in exported forensic CSVs.
3. **Hardcoded Data Classification:** `unified_timeline.csv` hardcodes `data_classification = "DERIVED_RESULT"` across every row, failing to distinguish measured physical facts, algorithmic estimates, and examiner-entered adjustments as required by forensic standards.
4. **Git Custody Leakage:** `.gitignore` specifies `evidence/` instead of `data/` or `data/evidence/`. As verified by `git status`, ingested video evidence files and audit logs remain untracked files in `data/`, creating a severe risk of accidental leak of private evidence to GitHub.
5. **Uploaded File Orphanage & Candidate Matching Disconnection:** Uploaded video files leave unvalidated temporary files on disk forever, and the application never triggers candidate event matching on user-uploaded videos (matching only executes during the synthetic demo).

---

## Existing Test Results

The existing test suite was executed in the active Python 3.11 environment using `python -m pytest -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\krish\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\krish\OneDrive\Desktop\Krishna\Hit and run case
plugins: anyio-4.15.1, platformdirs-4.13.5
collected 23 items

tests/test_evidence_integrity.py::test_sha256_computation PASSED         [  4%]
tests/test_evidence_integrity.py::test_evidence_ingestion_and_vault PASSED [  8%]
tests/test_evidence_integrity.py::test_tamper_detection PASSED           [ 13%]
tests/test_evidence_integrity.py::test_empty_evidence_rejection PASSED   [ 17%]
tests/test_exports.py::test_export_all_artifacts PASSED                  [ 21%]
tests/test_synthetic_end_to_end.py::test_end_to_end_forensic_pipeline PASSED [ 26%]
tests/test_time_aligner.py::test_reference_camera_calibration PASSED     [ 30%]
tests/test_time_aligner.py::test_insufficient_anchors PASSED             [ 34%]
tests/test_time_aligner.py::test_single_anchor_constant_offset PASSED    [ 39%]
tests/test_time_aligner.py::test_known_positive_and_negative_offsets_with_drift PASSED [ 43%]
tests/test_time_aligner.py::test_ransac_rejects_false_visual_matches PASSED [ 47%]
tests/test_time_aligner.py::test_conflicting_anchors_flagged_for_human_review PASSED [ 52%]
tests/test_time_aligner.py::test_reproducibility_of_estimated_offsets PASSED [ 56%]
tests/test_time_aligner.py::test_timing_error_metrics PASSED             [ 60%]
tests/test_video_metadata.py::test_valid_video_metadata_extraction PASSED [ 65%]
tests/test_video_metadata.py::test_nonexistent_file_handling PASSED      [ 69%]
tests/test_video_metadata.py::test_empty_zero_byte_video_handling PASSED [ 73%]
tests/test_video_metadata.py::test_corrupt_header_handling PASSED        [ 78%]
tests/test_visual_matcher.py::test_phash_and_dhash_on_identical_frames PASSED [ 82%]
tests/test_visual_matcher.py::test_phash_similarity_robust_to_slight_lighting PASSED [ 86%]
tests/test_visual_matcher.py::test_distinct_images_have_low_similarity PASSED [ 91%]
tests/test_visual_matcher.py::test_candidate_match_generation PASSED     [ 95%]
tests/test_visual_matcher.py::test_manual_anchor_creation PASSED         [100%]

============================= 23 passed in 3.80s ==============================
```

### Note on Test Execution Environment
Attempting to run `pytest -v` directly resulted in:
`Program 'pytest.exe' failed to run: An Application Control policy has blocked this file`.
Under Windows Application Control / AppLocker environments, standalone script entrypoints in Python Scripts directories can be restricted. Running through the Python runtime module `python -m pytest -v` succeeded completely and executed all 23 tests without failure.

---

## Section A: Critical Defects

### Finding A.1: False Reference Clock Assignment When Confirming Cross-Camera Matches
* **Associated File & Function:** [app.py:L601-L614](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L601-L614) (`tab_matches` anchor confirmation button handler)
* **Description:**
  When an investigator confirms a candidate match in the UI, `app.py` appends a `TimeAnchorPoint` with:
  ```python
  TimeAnchorPoint(
      match_id=m.match_id,
      camera_id=m.camera_b_id,
      source_time_sec=m.time_b_sec,
      reference_time_sec=m.time_a_sec
  )
  ```
  This logic blindly assumes that `m.camera_a_id` is always the designated Reference Timeline Camera.
  In a multi-camera case (e.g., Cam 1 is reference, but a match occurs between Cam 2 and Cam 3), `time_a_sec` is Cam 2's *uncalibrated local clock time*, NOT the global reference timeline. By setting `reference_time_sec = m.time_a_sec`, Cam 3 is aligned directly against Cam 2's uncorrected clock, introducing false timing offsets.
  Furthermore, if `m.camera_b_id` happens to be the reference camera, it creates an anchor for the reference camera itself, while leaving the other camera completely uncalibrated.
* **Forensic Impact:** Catastrophic timeline corruption. In multi-camera cases, non-reference correspondences induce erroneous clock alignments that invalidate any subsequent collision speed or event sequence reconstruction.
* **Recommended Fix:**
  1. Determine which camera in the match is the reference camera. If one is the reference camera ($Cam_{ref}$), the other camera ($Cam_{src}$) receives an anchor with $t_{src} = \text{local time}$ and $t_{ref} = \text{reference local time}$.
  2. If *neither* camera is the reference camera, check whether one camera is already synchronized. If $Cam_a$ is synchronized with mapping function $T_{ref}(t_a)$, then calculate $t_{ref} = T_{ref}(t_a)$ to map $Cam_b$ transitively to the reference timeline.
  3. If neither camera is yet anchored to the reference camera, present an explicit prompt or flag that direct reference anchoring is required before transitive calibration.
* **Verification Method:** Create a unit test where Cam 1 is reference ($offset = 0$), Cam 2 has offset $+3.5$s, and Cam 3 has offset $-2.0$s. Register an anchor between Cam 2 ($t=2.5$s) and Cam 3 ($t=8.0$s). Verify that Cam 3's calculated global offset matches ground truth rather than $-2.5$s.

---

### Finding A.2: Fabrication of `review_status = 'CONFIRMED'` in Timeline Events
* **Associated File & Function:** [app.py:L476-L480](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L476-L480) and [src/timeline_builder.py:L136-L179](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/timeline_builder.py#L136-L179)
* **Description:**
  In `app.py`:
  ```python
  unified_events = tb.build_unified_events(
      st.session_state.events,
      st.session_state.candidate_matches,  # <--- passed as confirmed_matches!
      st.session_state.candidate_matches   # <--- passed as candidate_matches!
  )
  ```
  In `TimelineBuilder.build_unified_events`, the method indexes all items in `confirmed_matches` into `confirmed_map`. If an event key is in `confirmed_map`, it sets:
  ```python
  rev_status = "CONFIRMED"
  ```
  Because `st.session_state.candidate_matches` (which contains unreviewed algorithmic candidate suggestions) was passed as `confirmed_matches`, *every* candidate match event is automatically assigned `review_status = "CONFIRMED"` in the unified timeline!
* **Forensic Impact:** Directly violates Digital Forensics Rule: *Do not fabricate evidence or review statuses*. Unreviewed heuristic visual matches appear in the official exported court report as human-confirmed events.
* **Recommended Fix:**
  In `app.py`, separate confirmed matches from unverified candidate matches:
  ```python
  confirmed_matches = [m for m in st.session_state.candidate_matches if m.review_status == "CONFIRMED_BY_USER"]
  unverified_candidates = [m for m in st.session_state.candidate_matches if m.review_status == "CANDIDATE_UNVERIFIED"]
  unified_events = tb.build_unified_events(
      st.session_state.events,
      confirmed_matches,
      unverified_candidates
  )
  ```
* **Verification Method:** Create unreviewed candidate matches. Call `build_unified_events` and assert that events not explicitly confirmed retain `review_status = "UNREVIEWED"` or `"CANDIDATE_UNVERIFIED"`.

---

### Finding A.3: Failure to Distinguish Measured Facts, Algorithmic Estimates, and Manual Overrides in Exports
* **Associated File & Function:** [src/report_exporter.py:L135](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/report_exporter.py#L135) (`export_unified_timeline_csv`)
* **Description:**
  In `ForensicReportExporter.export_unified_timeline_csv`, the `data_classification` field is hardcoded:
  ```python
  "data_classification": "DERIVED_RESULT"
  ```
  This applies `"DERIVED_RESULT"` indiscriminately to:
  - Original camera frame indices and source timestamps (which are raw measured physical facts).
  - RANSAC-interpolated timestamps (which are algorithmic estimates subject to uncertainty bounds).
  - Manual investigator adjustments (which are examiner-entered corrections).
* **Forensic Impact:** Inadmissibility risk under evidentiary rules (e.g., Federal Rule of Evidence 702 / Daubert standard). A forensic timeline must explicitly demarcate what was directly recorded by hardware vs. what was calculated by software vs. what was altered by the examiner.
* **Recommended Fix:**
  Compute `data_classification` dynamically per row:
  - If event is an unaltered camera recording point with unadjusted time: `"MEASURED_FACT"`.
  - If event timestamp is calculated via RANSAC linear regression: `"ALGORITHMIC_ESTIMATE"`.
  - If event timestamp or offset was manually overridden by the user: `"EXAMINER_OVERRIDE"`.
* **Verification Method:** Export a timeline containing a reference camera event, an aligned camera event, and a manually overridden event. Verify each row has the exact corresponding classification in the CSV output.

---

### Finding A.4: Git Custody Leakage — Video Evidence and Audit Logs Untracked
* **Associated File & Function:** [.gitignore:L33-L37](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/.gitignore#L33-L37)
* **Description:**
  `.gitignore` includes:
  ```text
  evidence_vault/
  evidence/
  reports/*
  !reports/README.md
  audit_log.jsonl
  ```
  However, the application stores evidence in `data/evidence/` and cache in `data/cache/`. Running `git status` reveals:
  ```text
  Untracked files:
    (use "git add <file>..." to include in what will be committed)
          data/
          sample_data/
  ```
  `data/evidence/` contains actual video files (`EVID-*.mp4`) and the append-only `audit_log.jsonl`. Because `data/` is not in `.gitignore`, running `git add .` or committing changes would commit real evidence files and internal audit logs directly into the public git repository.
* **Forensic Impact:** Severe violation of evidence privacy, confidentiality, and chain-of-custody protocols.
* **Recommended Fix:**
  Update `.gitignore` to explicitly ignore `data/`, `data/**`, `evidence_vault/`, `sample_data/*.mp4`, `*.mp4`, `*.avi`, `*.mov`, `*.mkv`, while retaining necessary documentation or sample placeholders.
* **Verification Method:** Run `git status` after updating `.gitignore` and ensure no video files or `data/` files appear under Untracked files.

---

### Finding A.5: User-Uploaded Video Pipeline Disconnected from Visual Matching
* **Associated File & Function:** [app.py:L284-L327](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L284-L327)
* **Description:**
  In `app.py`, when a user uploads real video files:
  1. Files are saved to `CACHE_DIR / uf.name`.
  2. Registered in `EvidenceManager`.
  3. Metadata extracted via `VideoMetadataExtractor`.
  4. Visual events detected via `EventDetector`.
  5. **However, `VisualMatcher.match_events()` is NEVER called for uploaded files!**
  In contrast, `load_demo_dataset()` calls `matcher.match_events(...)`. For uploaded files, `st.session_state.candidate_matches` remains permanently empty. There is no button, trigger, or background process in Tab 3 or elsewhere to generate candidate matches for uploaded videos.
* **Forensic Impact:** The application is functionally inoperable for real-world user-uploaded investigations unless the user manually types in raw numbers in Tab 4.
* **Recommended Fix:**
  Add an explicit matching step: after ingesting uploaded videos (or via a dedicated "Find Visual Correspondences" button in Tab 3), run pairwise `VisualMatcher.match_events()` across all active cameras and populate `st.session_state.candidate_matches`.
* **Verification Method:** Upload two test video files via the Streamlit interface or simulation, verify that candidate matches are generated and visible in Tab 3.

---

## Section B: High-Priority Defects

### Finding B.1: False Certainty in 2-Point Synchronization ($N=2$ Zero Degree-of-Freedom)
* **Associated File & Function:** [src/time_aligner.py:L225-L249](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/time_aligner.py#L225-L249)
* **Description:**
  When exactly two anchor points are provided for a camera, a 2-parameter linear model ($offset, rate$) fits the two points with zero degrees of freedom ($DoF = N - 2 = 0$). As a result, residuals are algebraically identical to zero ($RMSE = 0.0$).
  In `TimeAligner`:
  ```python
  status = "SYNCHRONIZED" if (rmse <= self.residual_threshold and len(rejected_ids) == 0) else "APPROXIMATE"
  ```
  Because $RMSE = 0 \le 0.35$ and $rejected\_ids = 0$, the alignment status is marked `"SYNCHRONIZED"`.
  Two points offer zero redundancy, zero capability to detect outliers, and zero statistical validation. If either anchor point is mistaken, the estimated clock drift rate will be completely wrong, yet the system reports `SYNCHRONIZED` with `RMSE: 0.0ms`.
* **Recommended Fix:**
  Require at least 3 independent anchor points ($N \ge 3$, providing $DoF \ge 1$) before granting `"SYNCHRONIZED"` status. When $N = 2$, set status to `"APPROXIMATE (UNVERIFIED_DRIFT)"` with an explicit forensic disclaimer stating that clock drift has no redundant validation.
* **Verification Method:** Supply 2 anchor points to `align_camera_from_anchors`. Assert that status is `APPROXIMATE` rather than `SYNCHRONIZED`.

---

### Finding B.2: Temporary File Leakage and Lack of Upload Validation
* **Associated File & Function:** [app.py:L301-L305](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L301-L305)
* **Description:**
  In `app.py`:
  ```python
  temp_path = CACHE_DIR / uf.name
  temp_path.write_bytes(uf.getbuffer())
  rec = em.register_evidence_file(temp_path, camera_name=cam_name)
  ```
  1. `uf.name` is directly concatenated without filename sanitization, opening path traversal or collision vulnerabilities.
  2. `temp_path` is written, copied to the evidence vault, but NEVER deleted. If a user uploads multiple multi-gigabyte video files, disk space will be rapidly exhausted.
  3. No validation of maximum file size, minimum file size, or file magic bytes. A 0-byte file causes `register_evidence_file` to throw an unhandled `ValueError`, crashing the Streamlit app session.
* **Recommended Fix:**
  1. Use Python's `tempfile.NamedTemporaryFile` with automatic cleanup in a `try...finally` block.
  2. Sanitize filenames using `Path(uf.name).name` with strict regex filtering (`[^a-zA-Z0-9_.-]`).
  3. Validate file size ($> 0$ and $< \text{MAX\_BYTES}$) and verify video container header magic bytes before ingestion.
* **Verification Method:** Upload a 0-byte file and an invalid header file; verify user-friendly error banners are displayed and no orphaned files remain in `CACHE_DIR`.

---

### Finding B.3: Windows File System Permission Failures Due to Read-Only Marking
* **Associated File & Function:** [src/evidence_integrity.py:L130-L133](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/evidence_integrity.py#L130-L133)
* **Description:**
  In `EvidenceManager.register_evidence_file`:
  ```python
  try:
      os.chmod(dest_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
  except Exception:
      pass
  ```
  On Windows, setting a file to `stat.S_IREAD` prevents any modification OR deletion by any process, including test cleanup (`tmp_path` teardown) or application cache clearing. Any subsequent attempt by `shutil.rmtree` or `os.remove` will raise `PermissionError: [WinError 5] Access is denied`.
* **Recommended Fix:**
  Provide an explicit vault cleanup/unlock utility that clears read-only attributes before deletion:
  ```python
  def unlock_file(path: Path):
      os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
  ```
* **Verification Method:** Write a test that ingests a file, marks it read-only, and verifies that vault teardown / cleanup cleans it up cleanly without `PermissionError`.

---

### Finding B.4: Path Invalidation on Camera Names with Windows Reserved Characters
* **Associated File & Function:** [src/evidence_integrity.py:L124](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/evidence_integrity.py#L124) and [src/frame_extractor.py:L169](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/frame_extractor.py#L169)
* **Description:**
  `vault_filename = f"{evidence_id}_{camera_name.replace(' ', '_')}{ext}"` and `filename = f"thumb_{camera_id}_f{frame_idx:06d}.jpg"`.
  If an investigator names a camera `Camera 1: Pole (North/West)` or `CAM:01`, the filename will contain `:`, `/`, or `\`. On Windows, these characters are illegal in filenames, causing `shutil.copy2` or `cv2.imwrite` to fail with `OSError: [Errno 22] Invalid argument`.
* **Recommended Fix:**
  Sanitize all camera identifiers using regex:
  ```python
  safe_camera_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', camera_name)
  ```
* **Verification Method:** Ingest a video with camera name `"Cam 1 / Traffic : North"`. Verify file is saved without `OSError`.

---

### Finding B.5: Silent Exception Suppression in Scene and Video Detection
* **Associated File & Function:** [src/event_detector.py:L131-L133](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/event_detector.py#L131-L133) and [src/video_metadata.py:L232-L233](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/video_metadata.py#L232-L233)
* **Description:**
  In `EventDetector._detect_pyscenedetect`:
  ```python
  except Exception as ex:
      # PySceneDetect fallback: continue gracefully
      pass
  ```
  And in `VideoMetadataExtractor._inspect_ffmpeg`:
  ```python
  except Exception:
      pass
  ```
  Both blocks catch and suppress all exceptions silently without logging, recording an audit log entry, or alerting the caller. If PySceneDetect crashes due to an unsupported codec or broken dependency, the user is given no indication that scene detection failed.
* **Recommended Fix:**
  Log the exception with traceback and attach an `error_message` or audit warning to the metadata / event detector output.
* **Verification Method:** Pass a corrupted video to `_detect_pyscenedetect` and assert that a structured warning/error is logged or returned rather than silently discarded.

---

## Section C: Medium-Priority Improvements

### Finding C.1: Unbounded In-Memory Frame Retention in `FrameExtractor.sample_frames`
* **Associated File & Function:** [src/frame_extractor.py:L90-L93](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/frame_extractor.py#L90-L93)
* **Description:**
  `SampledFrame` retains `image_bgr=frame.copy()`. Sampling 1,000 frames from a 1080p stream stores ~6.2 GB of uncompressed numpy arrays in memory. While `SampledFrame.release_image()` exists, it is never called.
* **Recommended Fix:**
  Do not keep raw BGR frames in memory if `save_thumbnails=True`. Only load frames on-demand from disk when needed.

---

### Finding C.2: Zero-Total-Frames Boundary Condition in Timeline Query Playhead
* **Associated File & Function:** [src/timeline_builder.py:L267](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/timeline_builder.py#L267)
* **Description:**
  `frame_idx = min(meta.total_frames - 1, max(0, int(round(src_time * fps))))`.
  If a video has `total_frames = 0` (unindexed stream or corrupt header), `total_frames - 1` is `-1`. The formula evaluates to `min(-1, ...)` yielding `-1`, which causes `cv2.CAP_PROP_POS_FRAMES` to fail.
* **Recommended Fix:**
  Guard against `meta.total_frames <= 0`:
  ```python
  max_frame = max(0, meta.total_frames - 1) if meta.total_frames > 0 else 0
  frame_idx = min(max_frame, max(0, int(round(src_time * fps))))
  ```

---

### Finding C.3: ISO Datetime Parser Rejects Valid ISO8601 Subsecond Timestamps
* **Associated File & Function:** [src/video_metadata.py:L243-L250](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/video_metadata.py#L243-L250)
* **Description:**
  The hardcoded format list in `_parse_iso_datetime` lacks `%Y-%m-%dT%H:%M:%S.%f` (fractional seconds without explicit timezone offset). Container timestamps like `2026-10-26T12:30:15.500` fall through unparsed.
* **Recommended Fix:**
  Use Python 3.11's standard `datetime.fromisoformat()` with fallback.

---

### Finding C.4: Omission of Investigator Justification in Manual Clock Overrides
* **Associated File & Function:** [app.py:L641-L666](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L641-L666)
* **Description:**
  In Tab 4 ("Manual Camera Offset Override"), examiners can change the offset and clock rate, but there is no text field for entering a forensic rationale or justification. The audit log records only the numeric change.
* **Recommended Fix:**
  Require a mandatory `Override Reason / Basis` text field before enabling the "Apply Manual Calibration" button.

---

### Finding C.5: Single-Point Intercept Uncertainty Overstatement ($se_{\text{offset}}$)
* **Associated File & Function:** [src/time_aligner.py:L218-L224](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/time_aligner.py#L218-L224)
* **Description:**
  `se_offset = rmse * np.sqrt(1.0 / n_inliers + (np.mean(inlier_src)**2) / s_xx)`.
  If anchor events occur late in a long video (e.g. $t \approx 1000$s), $\text{mean}(t)^2 / s_{xx}$ causes the standard error of the intercept at $t=0$ to blow up, even though the alignment uncertainty near the actual anchor points is minimal. Reporting this scalar as the constant uncertainty across the entire timeline is misleading.
* **Recommended Fix:**
  Report uncertainty as a function of query playhead time $t$, or report the residual RMSE and maximum anchor span explicitly.

---

## Section D: Deployment Problems

| Issue | Impact | Environment Affected | Recommended Resolution |
| :--- | :--- | :--- | :--- |
| **Windows Application Control Policy** | Running `pytest` or CLI entry points directly fails with `NativeCommandFailed` | Windows 10/11 enterprise or restricted environments | Always invoke tools via the python runtime: `python -m pytest` and `python -m streamlit run app.py`. Document this in `README.md`. |
| **Missing Linux Shared Libraries (`packages.txt`)** | OpenCV (`cv2`) and PySceneDetect require `libgl1`, `libglib2.0-0` on Debian/Ubuntu/Streamlit Cloud | Linux containers, Docker, Streamlit Cloud | Create `packages.txt` containing `libgl1` and `libglib2.0-0` for Streamlit Cloud deployment. |
| **Ephemeral Local Storage in Cloud Hosting** | Vault videos in `data/evidence` and reports in `reports/` are wiped on container restart | Streamlit Cloud / HuggingFace Spaces / Heroku | Support an optional external storage directory or clarify local workstation deployment requirements. |
| **Concurrent Multi-User State Contamination** | Shared `data/cache` and `data/evidence` across all browser sessions if hosted centrally | Multi-user Streamlit deployments | Scope cache directories to unique forensic case IDs or session identifiers. |
| **Dependency Pinning Fragility** | `requirements.txt` has strictly pinned versions (e.g., `numpy==2.4.6`, `scipy==1.17.1`) | Future virtual environments | Use compatible range operators (`numpy>=2.0,<3.0`) or maintain separate `requirements.txt` (flexible) and `requirements-lock.txt` (pinned). |

---

## Section E: Missing Tests

The existing test suite has 23 passing tests, but major subsystems and edge cases have zero test coverage:

1. **`FrameExtractor` Tests (0% coverage):**
   - No test for `sample_frames` uniform time step extraction.
   - No test for `extract_frame_at_timestamp` seeking and boundary fallback.
   - No test for `extract_frame_at_index` bounds checking.
   - No test for thumbnail generation and caching.
2. **`EventDetector` Unit Tests (0% direct unit coverage):**
   - No unit test for `detect_events` with static video (zero cuts/motion).
   - No unit test verifying motion spike thresholding against known noise.
   - No unit test for PySceneDetect fallback on corrupted frames.
3. **`TimelineBuilder` Unit Tests (0% direct unit coverage):**
   - No unit test for `compute_camera_spans` with mixed aligned and unaligned streams.
   - No unit test for `query_global_time` before recording start, during recording, and after recording end.
   - No unit test for handling `total_frames == 0`.
4. **`TimeAligner` Edge Cases:**
   - No test for $N=2$ anchors verifying that zero DoF is not falsely reported as authoritative.
   - No test for anchors separated by $< 0.2$s.
   - No test for manual override preservation in alignment calculations.
5. **Report Classification & Export Safety:**
   - No test verifying dynamic `data_classification` (`MEASURED_FACT` vs `ALGORITHMIC_ESTIMATE` vs `EXAMINER_OVERRIDE`).
   - No test checking that `NaN` or `Infinity` floats are safely converted in JSON exports without breaking RFC 8259 compliance.
6. **Filename & Path Security:**
   - No test verifying that camera names with `/`, `\`, or `:` are sanitized without crashing on Windows.

---

## Section F: Summary of Findings by Exact File and Function

| Finding ID | Priority | File | Function / Location | Summary |
| :--- | :--- | :--- | :--- | :--- |
| **A.1** | Critical | [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L601-L614) | Tab 3 Anchor Confirm Button | Non-reference match confirms falsely assign local time as reference timeline time |
| **A.2** | Critical | [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L476-L480) & [src/timeline_builder.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/timeline_builder.py#L136-L179) | `build_unified_events` | Candidate matches passed as confirmed, fabricating `CONFIRMED` review status in reports |
| **A.3** | Critical | [src/report_exporter.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/report_exporter.py#L135) | `export_unified_timeline_csv` | Hardcoded `data_classification = "DERIVED_RESULT"` across all rows |
| **A.4** | Critical | [.gitignore](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/.gitignore#L33-L37) | Root `.gitignore` | `data/` and `sample_data/` untracked, risking accidental evidence commits to git |
| **A.5** | Critical | [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L284-L327) | Evidence Ingestion Handler | Uploaded video files never trigger visual matching; matching only runs in demo |
| **B.1** | High | [src/time_aligner.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/time_aligner.py#L225-L249) | `align_camera_from_anchors` | 2-point alignment ($DoF=0$) falsely reports authoritative `SYNCHRONIZED` status |
| **B.2** | High | [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L301-L305) | Upload Processing | Temporary files leaked in `data/cache`; no size or header validation |
| **B.3** | High | [src/evidence_integrity.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/evidence_integrity.py#L130-L133) | `register_evidence_file` | Read-only attribute causes Windows `PermissionError` on cleanup/deletion |
| **B.4** | High | [src/evidence_integrity.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/evidence_integrity.py#L124) | `register_evidence_file` | Camera names with reserved characters (`:`, `/`, `\`) crash on Windows |
| **B.5** | High | [src/event_detector.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/event_detector.py#L131-L133) | `_detect_pyscenedetect` | Silent exception suppression hides detector failures |
| **C.1** | Medium | [src/frame_extractor.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/frame_extractor.py#L90-L93) | `sample_frames` | Large uncompressed BGR frames retained in memory |
| **C.2** | Medium | [src/timeline_builder.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/timeline_builder.py#L267) | `query_global_time` | Zero total frames results in `-1` frame index |
| **C.3** | Medium | [src/video_metadata.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/video_metadata.py#L243-L250) | `_parse_iso_datetime` | Rejects ISO8601 subsecond timestamps without timezone |
| **C.4** | Medium | [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L641-L666) | Tab 4 Manual Override | Manual offset overrides lack rationale/justification entry in audit log |
| **C.5** | Medium | [src/time_aligner.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/time_aligner.py#L218-L224) | `align_camera_from_anchors` | Single-point $se_{\text{offset}}$ overstates timeline uncertainty for late events |
| **D.1** | Deployment | [packages.txt](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/packages.txt) | Root Configuration | Missing `packages.txt` for Linux/Streamlit Cloud dependencies (`libgl1`, `libglib2.0-0`) |

---

## Section G: Prioritized Remediation Plan

### Step 1: Protect Evidence Privacy & Git Custody (Immediate)
* Update [.gitignore](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/.gitignore) to exclude `data/`, `data/**`, `sample_data/*.mp4`, `*.tmp`, `*.part`.
* Verify with `git status` that no evidence files are staged or untracked.

### Step 2: Fix Core Evidentiary & Synchronization Algorithms
* Fix [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L601-L614) to resolve reference camera mapping transitively when confirming anchors between non-reference cameras.
* Separate confirmed matches from candidate matches in [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py#L476-L480) to stop fabricating `review_status = "CONFIRMED"` on unreviewed events.
* Update [src/report_exporter.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/report_exporter.py#L135) to dynamically classify rows as `MEASURED_FACT`, `ALGORITHMIC_ESTIMATE`, or `EXAMINER_OVERRIDE`.
* Require $N \ge 3$ for `SYNCHRONIZED` status in [src/time_aligner.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/time_aligner.py#L225-L249).

### Step 3: Connect Uploaded Video Workflow & Safe File Ingestion
* Connect [app.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/app.py) upload handler to pairwise `VisualMatcher.match_events()`.
* Add temporary file cleanup and file size/header validation during upload.
* Sanitize camera names to prevent Windows reserved character errors in [src/evidence_integrity.py](file:///c:/Users/krish/OneDrive/Desktop/Krishna/Hit%20and%20run%20case/src/evidence_integrity.py).

### Step 4: Expand Automated Test Suite
* Add comprehensive tests for `FrameExtractor`, `EventDetector`, `TimelineBuilder`, transitive cross-camera anchoring, and report classification.
* Verify 100% test pass rate via `python -m pytest -v`.

---

*Report authored and filed under docs/TECHNICAL_AUDIT.md in accordance with Phase 1 instructions.*
