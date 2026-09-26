"""Keep dense DINO features registered with the full RGB-D raster."""


def configure_full_frame_features(processor):
    """Use every raster pixel in the 16×16 patch grid expected by MTU3D.

    Upstream samples this grid with coordinates covering the whole depth map.
    A center crop would remove border pixels and misregister those coordinates.
    The benchmark's square camera permits a full-frame 224² resize without
    changing its aspect ratio. Normalization and interpolation stay unchanged.
    """
    processor.do_center_crop = False
    processor.size = {'height': 224, 'width': 224}
    return processor
