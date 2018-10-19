import numpy as np


def generate_anchors_reference(base_size, aspect_ratios, scales):
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = np.reshape(scales_grid, (-1))
    base_aspect_ratios = np.reshape(aspect_ratios_grid, (-1))
    print(base_scales)
    print(base_aspect_ratios)
    sqrt_aspect_ratios = np.sqrt(base_aspect_ratios)
    heights = base_size * base_scales * sqrt_aspect_ratios
    widths = base_size * base_scales / sqrt_aspect_ratios

    center_xy = 0

    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2
    ])

    return anchors
