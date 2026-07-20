# Front-Rim Benchmark

The front-rim estimator remains disabled in production until it passes the
offline and live qualification gates.

## Automatic opening-plane reference

`benchmarks/front_rim_ground_truth.json` is generated automatically in Isaac
Sim and is used only to score the physical front-opening plane. Production
modules must never import or read it.

Run:

```bash
$HOME/isaacsim/python.sh tools/extract_front_rim_ground_truth.py
```

The tool performs the entire calculation without user input:

1. YOLOE selects the port in both stereo eyes.
2. The dense rim estimator fits the visible front opening.
3. Corresponding rim samples are shifted outward onto the solid bezel.
4. Virtual-camera PhysX rays are cast through those bezel-ring pixels.
5. Rack hits are depth-clustered so recessed cavity hits are rejected.
6. A robust camera-facing bezel plane is fitted.
7. The detected opening-center ray is intersected with that plane.
8. Schema-version-2 JSON is written with the reference center, normal, plane
   residual, hit counts, and used rack prim paths.

There is no transform editing, authored guide, depth offset, or Enter-to-save
step. The raycast result is validation-only and cannot influence robot motion.

## Geometry tests

```bash
$HOME/isaacsim/python.sh -m unittest -v \
  tests.test_automatic_port_ground_truth \
  tests.test_front_rim_2d \
  tests.test_front_rim_stereo
```

## Frozen-frame benchmark gates

- pair success rate: at least 95%
- target switches: 0
- radial 3D jitter: at most 0.5 mm
- ray-gap p95: at most 0.5 mm
- median opening-plane error: at most 0.5 mm
- p95 opening-plane error: at most 1.0 mm

A failed gate leaves `CONFIG.front_rim.enabled=False`. Do not weaken the gate
or add an automatic cavity-center fallback.
