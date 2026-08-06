#!/usr/bin/env python3
"""Click-and-drag bounding box labeler for building the YOLOE reference
prompt. Run against one or more saved yoloe_ref_*.png images; drag a box
around the connector head in each, and the pixel XYXY coordinates print
to the console when you release the mouse. Close the window to move to
the next image.

Usage:
    python label_reference_box.py camera_output/yoloe_ref_left_0000.png camera_output/yoloe_ref_left_0020.png ...
"""

import sys

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Qt5Agg")  # fallback if tkinter / TkAgg isn't available

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image


def label_image(image_path: str) -> tuple[float, float, float, float] | None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    box: list[float] = []

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"{image_path}\nDrag a box around the head, then close this window")

    def on_select(eclick, erelease) -> None:
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        box[:] = [x1, y1, x2, y2]
        print(f"  box (xyxy): ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=3,
        minspany=3,
        spancoords="pixels",
        interactive=True,
    )
    plt.show()

    return tuple(box) if box else None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python label_reference_box.py <image1.png> [image2.png ...]")
        sys.exit(1)

    results: dict[str, tuple[float, float, float, float]] = {}
    for path in sys.argv[1:]:
        print(f"\n{path}")
        box = label_image(path)
        if box is None:
            print("  (no box drawn, skipped)")
            continue
        results[path] = box

    print("\n--- reference_boxes_xyxy candidates ---")
    for path, box in results.items():
        print(f"# {path}")
        print(f"({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}),")


if __name__ == "__main__":
    main()
