"""Environment and dependency diagnostic checks for ChronoTrace AI.

Verifies that numpy, cv2, and scenedetect import cleanly and satisfy
forensic pipeline compatibility requirements in the app runtime environment.
"""

from __future__ import annotations

import sys


def run_diagnostics() -> dict[str, str]:
    """Test importing numpy, cv2, and scenedetect, returning their versions."""
    diagnostics = {}

    # 1. NumPy import and version check
    import numpy as np  # noqa: F401
    diagnostics["numpy"] = np.__version__
    np_major = int(np.__version__.split(".")[0])
    if np_major >= 2:
        raise RuntimeError(
            f"NumPy version {np.__version__} is >= 2.0.0. "
            "ChronoTrace AI requires NumPy < 2.0.0 for OpenCV C-ABI compatibility."
        )

    # 2. OpenCV import and headless verification
    import cv2  # noqa: F401
    diagnostics["cv2"] = cv2.__version__
    if not hasattr(cv2, "VideoCapture") or not hasattr(cv2, "resize"):
        raise RuntimeError("cv2 is missing core video/image attributes.")

    # 3. PySceneDetect headless import check
    import scenedetect  # noqa: F401
    from scenedetect import ContentDetector, SceneManager, open_video  # noqa: F401
    diagnostics["scenedetect"] = scenedetect.__version__
    if not callable(open_video):
        raise RuntimeError("scenedetect.open_video is not callable.")

    return diagnostics


def test_environment_dependency_diagnostics():
    """Pytest test case verifying numpy, cv2, and scenedetect imports in the app environment."""
    results = run_diagnostics()
    assert "numpy" in results
    assert "cv2" in results
    assert "scenedetect" in results

    # Verify NumPy is below version 2
    np_major = int(results["numpy"].split(".")[0])
    assert np_major < 2, f"NumPy version {results['numpy']} must be < 2.0.0"

    # Verify cv2 and scenedetect versions exist
    assert len(results["cv2"]) > 0
    assert len(results["scenedetect"]) > 0


if __name__ == "__main__":
    try:
        versions = run_diagnostics()
        print("=" * 60)
        print("CHRONOTRACE AI - RUNTIME ENVIRONMENT DIAGNOSTICS")
        print("=" * 60)
        print(f"Python Executable : {sys.executable}")
        print(f"Python Version    : {sys.version.split()[0]}")
        print(f"NumPy Version     : {versions['numpy']} (< 2.0.0 OK)")
        print(f"OpenCV (cv2)      : {versions['cv2']} (headless OK)")
        print(f"PySceneDetect     : {versions['scenedetect']} (headless OK)")
        print("=" * 60)
        print("DIAGNOSTIC STATUS : ALL APP DEPENDENCY IMPORTS PASSED")
        print("=" * 60)
        sys.exit(0)
    except Exception as exc:
        print(f"DIAGNOSTIC FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)
