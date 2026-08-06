#!/usr/bin/env python3
"""Composite several labeled reference crops into one atlas image, and
print the translated box coordinates for CableHeadYOLOEConfig."""

from PIL import Image

# (image_path, box_xyxy) — box is in the ORIGINAL image's coordinates
ENTRIES = [
    ("camera_output/yoloe_ref_left_0010.png", (554, 224, 615, 318)),
    ("camera_output/yoloe_ref_left_0020.png", (501, 249, 565, 350)),
    ("camera_output/yoloe_ref_left_0030.png", (439, 280, 503, 391)),
    ("camera_output/yoloe_ref_left_0040.png", (374, 315, 434, 433)),
]

MARGIN = 20  # extra context pixels kept around each box

crops = []
for path, box in ENTRIES:
    img = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    cx1 = max(0, x1 - MARGIN)
    cy1 = max(0, y1 - MARGIN)
    cx2 = min(img.width, x2 + MARGIN)
    cy2 = min(img.height, y2 + MARGIN)
    crop = img.crop((cx1, cy1, cx2, cy2))
    local_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    crops.append((crop, local_box))

cell_w = max(c.width for c, _ in crops)
cell_h = max(c.height for c, _ in crops)
atlas = Image.new("RGB", (cell_w * 2, cell_h * 2), (128, 128, 128))

final_boxes = []
for i, (crop, local_box) in enumerate(crops):
    col, row = i % 2, i // 2
    ox, oy = col * cell_w, row * cell_h
    atlas.paste(crop, (ox, oy))
    lx1, ly1, lx2, ly2 = local_box
    final_boxes.append((ox + lx1, oy + ly1, ox + lx2, oy + ly2))

atlas.save("camera_output/yoloe_reference_head_atlas.png")

print("reference_boxes_xyxy = (")
for b in final_boxes:
    print(f"    ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}),")
print(")")
