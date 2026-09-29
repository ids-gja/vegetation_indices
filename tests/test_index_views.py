import numpy as np

from greenview_pro.processing import CameraWorker


def test_ndvi_and_cvi_use_distinct_channels_and_render_distinct_images():
    worker = CameraWorker()
    raw = np.array(
        [
            [5, 10, 10, 10],
            [30, 0, 30, 0],
            [20, 10, 40, 10],
            [30, 0, 30, 0],
        ],
        dtype=np.uint16,
    )
    indices = worker._raw_indices(raw)
    np.testing.assert_allclose(indices["ndvi"], np.full((2, 2), 5 / 9))
    np.testing.assert_allclose(
        indices["cvi"], [[896, 224 / 9], [224 / 64, 224 / 324]]
    )
    _, ndvi_image, cvi_image = worker._process(raw, 0)
    assert np.unique(ndvi_image.reshape(-1, 3), axis=0).shape[0] == 1
    assert np.unique(cvi_image.reshape(-1, 3), axis=0).shape[0] > 1
