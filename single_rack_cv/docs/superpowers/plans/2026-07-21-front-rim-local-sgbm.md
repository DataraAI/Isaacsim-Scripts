# Local SGBM Front-Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace failed sparse bezel matching in the frozen benchmark with local dense SGBM disparity and robust front-plane fitting.

**Architecture:** A new pure OpenCV module computes a narrow local disparity map, applies reverse-consistency filtering, extracts a four-sided bezel ring, triangulates calibrated stereo points, selects the nearest coherent range cluster, and fits the front plane. A separate benchmark implementation reuses existing detection, dataset, scoring, and report helpers while keeping the controller disabled.

**Tech Stack:** Python 3, NumPy, OpenCV StereoSGBM, existing `CameraModel`, existing frozen 60-pair benchmark.

## Global Constraints

- Ubuntu 24.04, Isaac Sim 6.0.0, ROS 2 Jazzy.
- Runtime estimation must be image-only.
- RTX/USD ground truth is benchmark scoring only.
- `CONFIG.front_rim.enabled` remains `False`.
- Existing qualification gates remain unchanged.
- Kill switch: stop local SGBM if pair success is below 80% or plane-error p95 exceeds 1.0 mm after one evidence-driven tuning pass.

---

### Task 1: Pure local disparity estimator

**Files:**
- Create: `single_rack_cv/front_rim_sgbm.py`
- Create: `single_rack_cv/tests/test_front_rim_sgbm.py`

**Interfaces:**
- Produces: `LocalSGBMConfig`, `LocalDisparityResult`, `compute_local_disparity(...)`.

- [ ] Write synthetic textured-shift tests for positive and negative disparity, flat-image rejection, and reverse consistency.
- [ ] Implement local crop construction, vertical alignment, forward/reverse SGBM, and consistency mask.
- [ ] Run `python -m unittest -v tests.test_front_rim_sgbm` and require all tests to pass.

### Task 2: Dense bezel-plane estimator

**Files:**
- Modify: `single_rack_cv/front_rim_sgbm.py`
- Modify: `single_rack_cv/tests/test_front_rim_sgbm.py`

**Interfaces:**
- Produces: `SGBMFrontPlaneResult`, `estimate_front_plane_sgbm(...)`.

- [ ] Add tests for four-sided ring construction and nearest coherent depth-cluster selection.
- [ ] Triangulate consistent ring pixels with calibrated rays and reject large ray gaps.
- [ ] Require support on all four sides, robustly fit the plane, orient the normal toward cameras, and intersect cavity-center rays.
- [ ] Run the SGBM and existing stereo tests.

### Task 3: Frozen 60-pair benchmark integration

**Files:**
- Create: `single_rack_cv/benchmarks/front_rim_sgbm_benchmark.py`
- Modify: `single_rack_cv/tools/run_front_rim_benchmark_isaac.py`
- Modify: `single_rack_cv/tools/run_front_rim_benchmark.sh`
- Modify: `single_rack_cv/tests/test_front_rim_benchmark.py`

**Interfaces:**
- Consumes: `estimate_front_plane_sgbm(...)`.
- Produces: the existing `details.csv`, `summary.json`, `report.txt`, and failure annotations with additional SGBM diagnostics.

- [ ] Add structural tests requiring the SGBM benchmark entry point and preserving summary-driven exit status.
- [ ] Reuse the existing YOLOE selection and truth-scoring helpers.
- [ ] Record disparity, consistency, triangulation, cluster, side-support, and plane metrics.
- [ ] Keep exit `0` for full qualification, `2` for completed-but-unqualified, and `1` for runtime failure.

### Task 4: Verification

**Files:**
- No new files.

- [ ] Run pure tests: `tests.test_front_rim_sgbm`, `tests.test_front_rim_2d`, `tests.test_front_rim_stereo`, `tests.test_front_rim_benchmark`.
- [ ] Run the frozen benchmark on the Isaac workstation.
- [ ] Honor the 80%/1 mm kill switch; do not weaken the gates.
