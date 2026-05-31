"""Small image helpers for SHAPify visual outputs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def draw_points_on_image(image_path, points, output_path, *, radius: int = 1, color=(0, 0, 255)) -> None:
    """Draw projected vertices on an image and write the result."""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
