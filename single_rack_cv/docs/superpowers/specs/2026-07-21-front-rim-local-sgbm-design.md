# Local SGBM Front-Plane Design

## Goal

Estimate the server-port front bezel plane from the existing 640x480 stereo RGB pair after YOLOE selects the correct dark cavity. Keep `CONFIG.front_rim.enabled=False` until the frozen 60-pair benchmark passes every existing qualification gate.

## Boundary

Runtime estimation is image-only. The RTX/USD ground-truth plane is used only after estimation to score benchmark error. It must never seed disparity, depth clustering, plane fitting, or center estimation.

## Pipeline

1. Use the selected left/right cavity detections and their refined mask centroids.
2. Vertically align the right image to the left image using the measured cavity-center vertical offset. Do not horizontally align; horizontal displacement is the stereo disparity signal.
3. Build a local crop spanning both detected cavities plus margin.
4. Run OpenCV StereoSGBM in a narrow disparity interval centered on the detected cavity-center disparity. Support positive or negative disparity with `minDisparity`.
5. Run a reverse right-to-left SGBM pass and retain pixels satisfying left-right consistency.
6. Form a narrow rectangular bezel ring outside the left cavity box and exclude cavity/interior pixels.
7. Triangulate valid dense ring pixels with the calibrated camera rays.
8. Reject points exceeding the ray-gap gate.
9. Find the densest coherent range cluster, breaking ties toward the nearest camera range, then require support on all four bezel sides.
10. Robustly fit the front plane, orient its normal toward the cameras, and intersect the two detected cavity-center rays with that plane.
11. Average the two center intersections only when they agree within the existing stereo gate.

## Diagnostics

For every frozen pair record valid disparity count, left-right-consistent count, triangulated point count, selected depth-cluster count, side support counts, disparity median, and plane residual. Save failed-frame annotations and a disparity visualization.

## Qualification and Kill Switch

The existing gates remain unchanged. If local SGBM cannot reach at least 80% pair success while keeping plane-error p95 at or below 1.0 mm, stop tuning this method. The next decision must be a sensor/configuration change: higher image resolution, larger stereo baseline, or active depth.
