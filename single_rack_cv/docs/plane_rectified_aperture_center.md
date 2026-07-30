# Plane-Rectified Physical Aperture Center

The live controller computes `/World/EstimatedPortPoint` from the physical insertion center of the stepped RJ45 opening after each eye contour is mapped onto the measured front-bezel plane.

The horizontal coordinate comes from the symmetry axis of the upper latch notch. The vertical coordinate is half the configured physical opening height above the visible bottom boundary. The current dimensions are the existing camera-independent RJ45 values: 11.4 mm wide and 7.0 mm high.

Runtime inputs are limited to synchronized RGB segmentation masks, calibrated camera models, the configured physical aperture dimensions, and the dense stereo front-plane fit. No empirical world-coordinate correction, rack transform, port prim, RTX hit, or USD ground truth is used.

The two eyes reconstruct the center independently. The frame is rejected when their reconstructed centers differ by more than 0.5 mm or when the rectified full width, stepped height, or latch-notch width is physically implausible.

The stage marker radius is 1 mm so its placement can be inspected without a port-sized sphere hiding the error.
