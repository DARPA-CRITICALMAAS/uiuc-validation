import os
import logging
import numpy as np
import rasterio
import multiprocessing

log = logging.getLogger('DARPA_CMAAS_VALIDATION')

def parallelLoadGeoTiffs(files, processes=1):
    p=multiprocessing.Pool()
    images = p.map(loadGeoTiff, files)
    p.close()
    p.join()

    return images

def loadGeoTiff(filepath):
    if not os.path.exists(filepath):
        log.warning('Image file "{}" does not exist. Skipping file'.format(filepath))
        return None
    with rasterio.open(filepath) as fh:
        image = fh.read()
        crs = fh.crs
        transform = fh.transform
    if image is None:
        log.warning('Could not load {}. Skipping file'.format(filepath))
        return None
    if len(image.shape) == 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

    return image, crs, transform

def saveGeoTiff(filename, prediction, crs=None, transform=None):
    """
    Save the prediction results to a specified filename.

    Parameters:
    - prediction: The prediction result (should be a 2D or 3D numpy array).
    - crs: The projection of the prediction.
    - transform: The transform of the prediction.
    - filename: The name of the file to save the prediction to.
    """

    if prediction.ndim == 3:
        image = prediction[...].transpose(2, 0, 1)  # rasterio expects bands first
    else:
        image = np.array(prediction[...], ndmin=3)
    rasterio.open(filename, 'w', driver='GTiff', compress='lzw',
                  height=image.shape[1], width=image.shape[2], count=image.shape[0], dtype=image.dtype,
                  crs=crs, transform=transform).write(image)

