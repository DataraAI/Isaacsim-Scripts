# Single-Rack Hard-Prune Design

## Objective

Reduce `single_rack_cv` to one understandable production path without changing the verified behavior of the 1280×960 automatic front-opening controller.

The cleanup must preserve:

- YOLOE cavity localization and tracking.
- Refined local SGBM front-plane estimation.
- Automatic opening-plane intersection with no manual depth offset.
- Translation-only stop-and-look control with a 1 mm maximum target step.
- Hold-on-failure behavior.
- No insertion command.
- The strict high-resolution benchmark and automatic RTX ground-truth workflow.

## Recovery Strategy

Before pruning, preserve the exact working state on:

`recovery/pre-single-rack-cleanup-2026-07-22`

The GitHub connector cannot create annotated tags, so the recovery branch is the remote recovery anchor. After the cleanup is merged, the local clone may additionally create an annotated tag with:

```bash
git tag -a single-rack-working-pre-cleanup-2026-07-22 \
  recovery/pre-single-rack-cleanup-2026-07-22 \
  -m "Working 1280x960 automatic front-plane controller before hard prune"
git push origin single-rack-working-pre-cleanup-2026-07-22
```

## Target Runtime Structure

```text
single_rack_cv/
├── main.py
├── config.py
├── sim.py
├── perception.py
├── debug.py
├── stereo_geometry.py
├── front_plane.py
├── live_control.py
├── benchmarks/
│   ├── capture_dataset.py
│   ├── front_plane_benchmark.py
│   └── front_plane_ground_truth.json
├── tools/
│   ├── run_benchmark.sh
│   └── generate_ground_truth.py
├── tests/
│   ├── test_front_plane.py
│   ├── test_live_control.py
│   ├── test_benchmark.py
│   └── test_ground_truth.py
└── README.md
```

Names may vary slightly where Isaac bootstrap requirements force a separate launcher, but there will be only one supported runtime and one supported benchmark path.

## Consolidation Rules

### Runtime

- Make 1280×960 the canonical camera configuration in `config.py`.
- Make automatic front-plane control the canonical behavior in `config.py`.
- Keep `main.py` as the only supported runtime command.
- Remove `main_highres.py` and `highres_config.py` after their behavior is folded into the canonical configuration.

### Front-plane estimation

- Move the final dense SGBM implementation and strict stabilized plane fit into `front_plane.py`.
- Move only reusable triangulation and ray helpers into `stereo_geometry.py`.
- Move the observation-replacement adapter and diagnostics into `live_control.py`.
- Delete the failed Sobel/rim and sparse patch-matching generations.
- Delete wrapper-on-wrapper modules after their final logic has been incorporated directly.

### Benchmark

- Collapse the layered benchmark chain into one final high-resolution benchmark module.
- Preserve all current qualification gates:
  - pair success rate ≥ 95%
  - track switches = 0
  - radial jitter ≤ 0.5 mm
  - ray-gap p95 ≤ 0.5 mm
  - plane-residual p95 ≤ 0.5 mm
  - plane-error median ≤ 0.5 mm
  - plane-error p95 ≤ 1.0 mm
- Keep dataset recapture and ground-truth regeneration automatic when resolution metadata is stale.
- Keep RTX/USD usage confined to benchmark ground truth; runtime remains image-only.

### Tests

- Preserve tests for final dense disparity, stabilized plane fitting, fused center-ray geometry, live observation replacement, hold-on-failure wiring, ground-truth metadata, and benchmark exit status.
- Delete tests whose only purpose is to preserve failed Sobel, sparse patch-matching, diagnostic, or obsolete wrapper behavior.
- Add a dependency-guard test that fails if production code imports deleted legacy modules.

### Documentation

- Replace historical working notes with one `README.md` containing:
  - architecture
  - exact runtime command
  - exact test command
  - exact benchmark command
  - qualification gates
  - safety constraints
  - recovery branch name
- Delete historical `docs/superpowers/plans/` and obsolete feature specs after the cleanup implementation is complete; Git history preserves them.

## Explicit Delete Candidates

Delete after dependency migration and tests pass:

- `front_rim.py`
- `front_rim_match.py`
- `front_rim_stereo.py`
- `front_rim_sgbm.py`
- `front_rim_sgbm_refined.py`
- `live_front_plane.py`
- `main_highres.py`
- `highres_config.py`
- `benchmarks/front_rim_benchmark_epipolar.py`
- `benchmarks/front_rim_sgbm_diagnostic.py`
- `benchmarks/front_rim_sgbm_benchmark.py`
- `benchmarks/front_rim_sgbm_refined_benchmark.py`
- `benchmarks/front_rim_sgbm_highres_benchmark.py`
- obsolete 2D, patch-matcher, and wrapper-specific tests
- historical implementation plans and superseded design specs

A file is deleted only after every surviving import has been migrated and the replacement tests pass.

## Error Handling

The cleanup must not weaken runtime failure handling:

- Detector failure: hold target and reacquire.
- SGBM failure: hold target and reacquire.
- Plane-fit failure: hold target and reacquire.
- No fallback to cavity depth.
- No fallback to a fixed recess offset.
- No insertion motion.

## Verification Gates

The cleanup is complete only when all of the following are true:

1. Pure unit tests pass.
2. Import/dependency guard passes.
3. The high-resolution frozen benchmark remains `QUALIFIED=true`.
4. `main.py` produces `[LIVE FRONT PLANE]` logs.
5. The marker remains on the front opening.
6. Final physical tracking error remains within the existing tolerance.
7. No DLSS minimum-resolution warning returns.
8. `camera_output/`, model weights, caches, and generated benchmark outputs remain ignored.
9. `git grep` finds no surviving imports of deleted legacy modules.

## Non-Goals

- No insertion behavior.
- No new detector model.
- No control-gain tuning.
- No qualification-gate changes.
- No camera-baseline or pose changes.
- No unrelated cleanup outside `single_rack_cv` except necessary root `.gitignore` updates.
