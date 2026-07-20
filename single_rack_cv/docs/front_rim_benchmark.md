# Front-Rim Benchmark

The front-rim estimator remains disabled in production until it passes every
offline and live qualification gate.

## Automatic opening-plane reference

`benchmarks/front_rim_ground_truth.json` is generated automatically in Isaac
Sim and is used only to score the physical front-opening plane. Production
modules must never import or read it.

Run the clean launcher:

```bash
bash tools/run_front_rim_ground_truth.sh
```

The tool performs the calculation without user input:

1. YOLOE selects the port in both stereo eyes.
2. The qualified dark-cavity result supplies only an approximate image region.
3. A rectangular ray ring is placed just outside the cavity onto the bezel.
4. RTX mesh rays intersect the visible USD geometry; no physics colliders are
   required.
5. Rack hits are depth-clustered so recessed geometry is rejected.
6. A robust camera-facing bezel plane is fitted.
7. The opening-center viewing ray is intersected with that plane.
8. Schema-version-3 JSON is written with the reference center, normal, plane
   residual, hit counts, and used USD mesh paths.

The result is validation-only and cannot influence robot motion.

## Geometry and metric tests

```bash
$HOME/isaacsim/python.sh -m unittest -v \
  tests.test_automatic_port_ground_truth \
  tests.test_front_rim_2d \
  tests.test_front_rim_stereo \
  tests.test_front_rim_benchmark
```

## Frozen 60-pair benchmark

The benchmark reruns the qualified Prompt B detector on the frozen stereo
capture, lets the production pairing logic select the port, then evaluates the
unqualified front-rim estimator independently.

Run:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv"
git pull --ff-only origin main
rm -rf camera_output/front_rim_benchmark_v1
bash tools/run_front_rim_benchmark.sh \
  2>&1 | tee camera_output/front_rim_benchmark_console.txt
```

The benchmark deliberately exits with status `2` when the estimator is not
qualified. That is a measured failure, not a launcher crash. Results are still
written to:

- `camera_output/front_rim_benchmark_v1/details.csv`
- `camera_output/front_rim_benchmark_v1/summary.json`
- `camera_output/front_rim_benchmark_v1/report.txt`
- `camera_output/front_rim_benchmark_v1/annotated/`

## Promotion gates

- pair success rate: at least 95%
- target switches: 0
- radial 3D jitter: at most 0.5 mm
- ray-gap p95: at most 0.5 mm
- median opening-plane error: at most 0.5 mm
- p95 opening-plane error: at most 1.0 mm

A failed gate leaves `CONFIG.front_rim.enabled=False`. Do not weaken a gate and
do not add an automatic cavity-center fallback.
