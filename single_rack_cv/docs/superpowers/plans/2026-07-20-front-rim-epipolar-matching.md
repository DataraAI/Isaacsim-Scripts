# Front-Rim Epipolar Patch Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified epipolar patch correspondences for cavity-anchored bezel samples before 3D triangulation.

**Architecture:** A new pure OpenCV matcher predicts each right-eye sample from cavity-center displacement, performs local normalized patch search with uniqueness and round-trip checks, and returns matched coordinates plus a validity mask. The existing triangulator consumes only those verified correspondences. The benchmark records match diagnostics and keeps production disabled until all gates pass.

**Tech Stack:** Python 3, NumPy, OpenCV, unittest, Isaac Sim 6.0.0 benchmark runner.

## Global Constraints

- Keep `CONFIG.front_rim.enabled=False`.
- Do not read RTX/USD ground truth from production modules.
- Do not change Prompt B detection or the active cavity-center controller.
- Preserve the existing qualification gates.
- Use no new third-party dependencies.

---

### Task 1: Pure epipolar patch matcher

**Files:**
- Create: `single_rack_cv/front_rim_match.py`
- Modify: `single_rack_cv/config.py`
- Create: `single_rack_cv/tests/test_front_rim_match.py`

**Interfaces:**
- Consumes: left/right RGB images, `FrontRim2D`, and `FrontRimConfig`.
- Produces: `EpipolarMatchResult` and `match_front_bezel_samples(...)`.

- [ ] Write tests for known 0.5-pixel-step translation recovery, low-texture rejection, and ambiguous repetitive-patch rejection.
- [ ] Run `python -m unittest -v tests.test_front_rim_match` and verify the tests fail because the matcher does not exist.
- [ ] Add matcher configuration fields for patch radius, search ranges, step, score, uniqueness, texture, round-trip error, and minimum accepted matches.
- [ ] Implement intensity/Sobel patch descriptors, normalized correlation, forward matching, backward matching, and result validation.
- [ ] Run the matcher tests and verify they pass.

### Task 2: Triangulator correspondence integration

**Files:**
- Modify: `single_rack_cv/front_rim_stereo.py`
- Modify: `single_rack_cv/tests/test_front_rim_stereo.py`

**Interfaces:**
- Consumes: optional `EpipolarMatchResult`.
- Produces: unchanged `FrontRim3D` output while triangulating only verified matches.

- [ ] Add a failing test where index-to-index right samples are wrong but explicit matched samples recover the plane.
- [ ] Extend `triangulate_front_rims(...)` with an optional match result.
- [ ] Initialize sample validity from the matcher mask and use its matched right coordinates.
- [ ] Preserve the legacy synthetic path when no matcher result is supplied.
- [ ] Run `tests.test_front_rim_stereo` and verify all tests pass.

### Task 3: Frozen benchmark integration and diagnostics

**Files:**
- Modify: `single_rack_cv/benchmarks/front_rim_benchmark.py`
- Modify: `single_rack_cv/tests/test_front_rim_benchmark.py`
- Modify: `single_rack_cv/tools/run_front_rim_benchmark.sh`

**Interfaces:**
- Consumes: selected left/right detections and matcher result.
- Produces: per-frame match count, score median, uniqueness median, and round-trip p95 in CSV/JSON/report output.

- [ ] Add a structural test requiring the benchmark to call `match_front_bezel_samples` and pass detection centroids into `extract_front_rim`.
- [ ] Invoke the matcher after cavity selection and before triangulation.
- [ ] Record matcher diagnostics even when later plane fitting rejects a frame.
- [ ] Update the launcher banner to `epipolar-patch-v4` while preserving summary-driven exit status.
- [ ] Run all pure front-rim tests.

### Task 4: Qualification run

**Files:**
- Generated only: `camera_output/front_rim_benchmark_v1/*`

- [ ] Pull the committed changes on the Isaac workstation.
- [ ] Run the pure unit-test suite.
- [ ] Run the frozen 60-pair benchmark.
- [ ] Accept the estimator only if every existing gate passes.
- [ ] If pair success remains below 80%, stop tuning and move to local SGBM design.
