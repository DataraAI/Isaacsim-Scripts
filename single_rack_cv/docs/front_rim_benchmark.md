# Front-Rim Benchmark

The front-rim estimator remains disabled in production until it passes the
offline and live qualification gates.

## Ground-truth calibration

`benchmarks/front_rim_ground_truth.json` is authored in Isaac Sim and is used
only to score the estimated physical front-opening plane. Production modules
must never import or read it.

Run:

```bash
$HOME/isaacsim/python.sh tools/calibrate_front_rim_ground_truth.py
```

The tool detects the current port, spawns and selects
`/World/PortOpeningGroundTruth`, and initializes it at the current recessed
cavity estimate. Translate the yellow guide plate outward until it is flush
with the physical front opening. Keep its red local +Z axis pointing toward
the stereo cameras. Press Enter in the terminal to write the JSON file.

## Frozen-frame benchmark gates

- pair success rate: at least 95%
- target switches: 0
- radial 3D jitter: at most 0.5 mm
- ray-gap p95: at most 0.5 mm
- median opening-plane error: at most 0.5 mm
- p95 opening-plane error: at most 1.0 mm

A failed gate leaves `CONFIG.front_rim.enabled=False`. Do not weaken the gate
or add an automatic cavity-center fallback.
