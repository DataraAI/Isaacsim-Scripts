# Plane-Rectified Center Qualification

The change is accepted only after an Isaac Sim workstation run proves that `/World/EstimatedPortPoint` lies at the physical center of the RJ45 opening from the angled wrist-camera view and the insertion stack remains inside its existing safety limits.

Required evidence:

- both eye masks are accepted without a manual correction vector;
- the 1 mm marker is visually centered in the physical aperture;
- stereo acquisition and handoff complete;
- all 48 insertion commands settle;
- lateral drift remains below 0.5 mm;
- orientation error remains below 1 degree;
- final depth remains approximately 10 mm inside the opening.

If the two plane-rectified eye centers disagree by more than 0.5 mm, the controller must hold and reacquire rather than guess.
