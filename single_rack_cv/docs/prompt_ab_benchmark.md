# YOLOE Prompt A/B Benchmark V1

This benchmark compares the working five-scale visual prompt atlas against one tight runtime-scale example using the **same 60 frozen stereo pairs**.

It adds standalone files only. It does not modify or replace:

- `main.py`
- `config.py`
- `perception.py`
- `sim.py`
- `debug.py`
- the geometry validation files

## Install

Benchmark source files live in:

```text
/home/aayush/Isaacsim-Scripts/single_rack_cv/benchmarks/
```

The `tests/` folder is optional at runtime but should be kept for verification.

## Run the complete benchmark

```bash
cd /home/aayush/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh benchmarks/run_prompt_ab_benchmark.py
```

The first child process starts Isaac Sim, holds the existing fixed startup target, waits for the ToolCenter to settle, and captures exactly 60 synchronized stereo pairs. It does not call the visual-servo observation method and therefore does not issue perception-based corrections.

The second child process does not start Isaac Sim. It evaluates:

- `A_five_scale_atlas`: all current prompt boxes
- `B_single_runtime_scale`: only the final atlas box, which is closest to the rendered 25–30 px port size

Each strategy gets one untimed warm-up. Timed inference then runs across all 60 identical frame pairs.

## Re-run evaluation without recapturing

```bash
~/isaacsim/python.sh benchmarks/run_prompt_ab_benchmark.py --reuse-frames
```

Use this only when the frozen frame set is unchanged and you want to rerun prompt evaluation.

## Outputs

```text
camera_output/prompt_ab_benchmark_v1/
├── manifest.json
├── frames/
│   ├── left_0001.png ... left_0060.png
│   └── right_0001.png ... right_0060.png
├── results/
│   ├── A_five_scale_atlas/
│   │   ├── details.csv
│   │   └── annotated/frame_0001.png ... frame_0060.png
│   └── B_single_runtime_scale/
│       ├── details.csv
│       └── annotated/frame_0001.png ... frame_0060.png
├── summary.csv
├── summary.json
└── winner.txt
```

## Qualification gates

A strategy qualifies only when all are true:

- stereo-pair success rate is at least 95%
- track-switch count is zero
- triangulated 3D center RMS jitter is at most 0.5 mm
- ray-gap p95 is at most 0.5 mm
- median inference time is no more than 25% slower than the faster strategy

The winner is selected among qualified strategies by:

1. higher stereo-pair success rate
2. fewer track switches
3. lower 3D jitter
4. lower median inference time

If `winner.txt` says `NO WINNER`, do not move the robot closer. Inspect each strategy's `details.csv` and annotated images first.

## Run unit and structural tests

```bash
cd /home/aayush/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest -v \
  tests.test_prompt_benchmark \
  tests.test_prompt_benchmark_structure
```
