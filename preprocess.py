"""
A *very* rough and dirty draft of a python translation of:

https://github.com/vazirilab/MAxiMuM_processing_tools :
Light Beads Microscopy with MAxiMuM Module: Data processing pipeline and work flow.

Python has a lot of advantages, namely the CaImAn, Suite2p and NormCorre libraries no
longer maintain their MATLAB versions.

This is both the start of a library to process LBM and general mesoscopic datasets, as well
as a way to learn more about the LBM data processing pipeline.

"""

from pathlib import Path
import json
import numpy as np
import util
from util import *
import ScanImageTiffReader

if __name__ == "__main__":
    home = Path.home()
    path = Path(home / "data" / "lbm" / "spon_6p45Hz_2mm_75pct_00001_00001.tif")
    with ScanImageTiffReader.ScanImageTiffReader(str(path)) as reader:
        metadata_str = reader.metadata()
        data = reader.data()

    # Deal with the non-json serializable string returned from the ScanImageTiffReader
    metadata_kv, metadata_json = util.parse(metadata_str)

    num_channels = 30
    num_volumes = data.shape[0] // num_channels
    num_rois = len(metadata_json["RoiGroups"]["imagingRoiGroup"]["rois"])

    # Reshape the data to be in the same format as the MATLAB code
    data = data.reshape(num_volumes, num_channels, *data.shape[1:]).transpose(
        2, 3, 1, 0
    )

    roi_data, roi_group = util.get_mroi_data_from_tiff(
        metadata_json, metadata_kv, data, num_channels, num_volumes, num_rois
    )

    num_rois = len(roi_data)
    totalFrame = len(roi_data[0].imageData[0])  # num_frames, same as num_volumes
    totalChannel = len(roi_data[0].imageData)  # num_channels, same as num_planes
    frameRate = metadata_kv["SI.hRoiManager.scanVolumeRate"]
    sizeXY = roi_group["rois"][0]["scanfields"]["sizeXY"]
    FOV = np.round(157.5 * np.array(sizeXY), 0).astype(int)
    numPX = roi_group["rois"][0]["scanfields"]["pixelResolutionXY"]
    pixelResolution = np.mean(np.divide(FOV, numPX)).round(3).astype(float)

    imageData = util.reorganize(numPX, totalChannel, num_rois, totalFrame, roi_data)

x = 4
