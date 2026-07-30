# Plane-Rectified Aperture Center

The live controller now computes `/World/EstimatedPortPoint` from the physical area centroid of the stepped RJ45 aperture after each eye mask is rectified onto the measured front-bezel plane.

Runtime inputs are limited to synchronized RGB segmentation masks, calibrated camera models, and the dense stereo front-plane fit. No empirical world-coordinate correction or rack/port ground truth is used.

The two eyes reconstruct the center independently. The frame is rejected when their reconstructed centers differ by more than 0.5 mm.

The stage marker radius is 1 mm so its placement can be inspected without a port-sized sphere hiding the error.
