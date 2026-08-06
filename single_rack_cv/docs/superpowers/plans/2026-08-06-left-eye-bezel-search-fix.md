# Left-eye outer-bezel search fix

## Failure evidence

The August 6 main-branch run accepted only capture 62. Most captures failed because the independent plane-rectified eye centers disagreed by 1–2.8 mm. The saved left-eye fit selected a farther outer-bezel edge, while the right-eye fit selected the physical lower-mouth wall.

## Root cause

`VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0114` caused the fitter to search 45% of that span, or 5.13 mm, outside each semantic-mask wall. The left-eye search therefore included a same-polarity outer-bezel shadow and selected it before the physical mouth edge.

## Fix

Set `VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0050`, producing a 2.25 mm exterior search radius. This retains the observed semantic-mask under-coverage while excluding the farther false bezel edge.

Unchanged:

- visible-width validation prior
- stereo center-disagreement gate: 0.5 mm
- opposite-edge parallelism gate
- edge reprojection gate
- camera transforms and rectification
- handoff and insertion controllers
- insertion calibration vector

## Verification

A synthetic reproduction matching the observed geometry changes the selected left edge from x=33 px with the old search span to x=94 px with the new span. The recovered mouth width is 12.70 mm instead of the false 15.75 mm bezel-to-wall width.

The Isaac Sim workstation run remains the final validation gate.
