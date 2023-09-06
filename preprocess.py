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

from util import parse_key_value
from util.return_scan_offset import return_scan_offset
from util.fix_scan_phase import fix_scan_phase
from util.roi_data_simple import RoiDataSimple
import ScanImageTiffReader

if __name__ == "__main__":
    home = Path.home()
    path = Path(home / 'data' / 'LBM' / 'spon_6p45Hz_2mm_75pct_00001_00001.tif')
    with ScanImageTiffReader.ScanImageTiffReader(str(path)) as reader:
        metadata_str = reader.metadata()
        data = reader.data()
    lines = metadata_str.split('\n')
    metadata_kv = {}
    json_portion = []
    parsing_json = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('SI.'):
            key, value = parse_key_value(line)
            metadata_kv[key] = value
        elif line.startswith('{'):
            parsing_json = True
        if parsing_json:
            json_portion.append(line)
    metadata_json = json.loads('\n'.join(json_portion))
    del line, key, value, metadata_str

    num_channels = 30
    num_volumes = data.shape[0] // num_channels
    num_rois = len(metadata_json['RoiGroups']['imagingRoiGroup']['rois'])

    # Reshape the data to be in the same format as the MATLAB code
    data = data.reshape(num_volumes, num_channels, *data.shape[1:]) \
        .transpose(2, 3, 1, 0)

    numLinesBetweenScanfields = np.round(
        metadata_kv['SI.hScan2D.flytoTimePerScanfield'] / float(metadata_kv['SI.hRoiManager.linePeriod']), 0)

    #  For us, always 0 ? Correspondes to number of slices in the stack
    stackZsAvailable = metadata_kv['SI.hStackManager.zs']
    if isinstance(stackZsAvailable, (int, float)):
        stackZsAvailable = 1
    else:
        lenRoiZs = len(stackZsAvailable)

    # roi_info is a little more complicated in python with array concatenation
    # Each stackZsAvailable increment will increase the length of the axis by 1
    # We don't actually need to handle this because in our case, it is 0 (or a single value)
    # So not worrying about it now: roi_info = np.zeros((num_rois, stackZsAvailable))  # MATLAB Translation
    roi_info = np.zeros((num_rois,))
    roi_img_height_info = np.zeros((num_rois, stackZsAvailable))

    num_image_categories = 1
    roi_data = {}
    roi_group = metadata_json['RoiGroups']['imagingRoiGroup']
    for i in range(num_rois):
        rdata = RoiDataSimple()
        rdata.hRoi = roi_group['rois'][i]
        rdata.channels = metadata_kv['SI.hChannels.channelSave']

        if isinstance(rdata.hRoi['zs'], (int, float)):
            lenRoiZs = 1
        else:
            lenRoiZs = len(rdata.hRoi['zs'])
        zsHasRoi = np.zeros_like(stackZsAvailable, dtype=int)  # This should likely go outside the for-loop

        if lenRoiZs == 1:
            if rdata.hRoi['discretePlaneMode']:
                zsHasRoi = (stackZsAvailable == rdata.hRoi['zs'][0]).astype(int)
                roi_img_height_info[i, np.where(zsHasRoi == 1)] = rdata.hRoi['scanfields']['pixelResolutionXY'][1]
            else:
                # The roi extends from -Inf to Inf
                zsHasRoi = np.ones_like(stackZsAvailable, dtype=int)
                # The height doesn't change for the case of single-scanfields
                roi_img_height_info[i, :] = rdata.hRoi['scanfields']['pixelResolutionXY'][1]
        else:
            minVal = rdata.hRoi['zs'][0]
            maxVal = rdata.hRoi['zs'][-1]
            idxRange = np.where((stackZsAvailable >= minVal) & (stackZsAvailable <= maxVal))[0]

            for j in idxRange:
                s = j
                sf = rdata.hRoi.get(stackZsAvailable[s])
                if sf:
                    roi_img_height_info[i, s] = sf['pixelResolution'][1]
                    zsHasRoi[s] = 1

        # We only need to fill in one row, this likely won't work for multiple zs
        roi_info[:] = zsHasRoi
        try:
            rdata.zs = np.array(stackZsAvailable)[np.where(zsHasRoi == 1)]
        except IndexError:
            rdata.zs = 0
        roi_data[i] = rdata

    for curr_channel in range(num_channels):
        for curr_volume in range(num_volumes):
            roi_image_cnt = np.zeros(num_rois, dtype=int)
            num_slices = [1]  # Placeholder
            for curr_slice in num_slices:
                num_curr_image_rois = np.sum(roi_info).astype(int)  # Adjust for other zs values
                roi_ids = np.where(roi_info[:] == 1)[0]

                cnt = 1
                prev_roi_img_height = 0
                img_offset_x, img_offset_y, roi_img_height = None, None, None
                for roi_idx in roi_ids:
                    if cnt == 1:
                        # The first one will be at the very top
                        img_offset_x = 0
                        img_offset_y = 0
                    else:
                        # For the rest of the rois, there will be a recurring numLinesBetweenScanfields spacing
                        img_offset_y = img_offset_y + roi_img_height + numLinesBetweenScanfields

                    # The width of the scanfield doesn't change
                    roi_img_width = roi_data[roi_idx].hRoi['scanfields']['pixelResolutionXY'][0]
                    # The height of the scanfield depends on the interpolation of scanfields within existing fields
                    roi_img_height = roi_img_height_info[roi_idx].astype(int)
                    # Need to account for array shapes when indexing with numpy (ugh)
                    roi_img_height_range = np.arange(0, roi_img_height)[:, None]
                    roi_img_width_range = np.arange(0, roi_img_width)
                    roi_image_cnt[roi_idx] += 1

                    # We need to ensure these indices aren't converted to floats
                    y_indices = (img_offset_y + roi_img_height_range).astype(int)
                    x_indices = (img_offset_x + roi_img_width_range).astype(int)

                    extracted_data = data[
                        y_indices,
                        x_indices,
                        curr_channel,
                        curr_volume
                    ]
                    roi_data[roi_idx].add_image_to_volume(curr_channel, curr_volume, extracted_data)
                    cnt += 1

    # roi_data - len(roi_data) = number of ROIs
    #  - RoiData Object (class)
    #     - zs (z stack, 0 for us)
    #     - channels (channels, the same for each ROI)
    #     - imageData (container)
    #       - len(imageData) = number of planes/channels (z)
    #         - len(imageData[0]) = number of frames/volumes (time)
    #           - len(imageData[0][0]) = number of "slices", but this will always be 1 for us
    #             - imageData[0][0][0] = the actual 2D cross-section of the image

    # Lots of redundant variables due to the ScanImage calculations done manually here
    # Keeping for comparison with MATLAB code
    num_rois = len(roi_data)
    totalFrame = len(roi_data[0].imageData[0])  # num_frames, same as num_volumes
    totalChannel = len(roi_data[0].imageData)  # num_channels, same as num_planes
    frameRate = metadata_kv['SI.hRoiManager.scanVolumeRate']
    sizeXY = roi_group['rois'][0]['scanfields']['sizeXY']
    FOV = np.round(157.5 * np.array(sizeXY), 0).astype(int)
    numPX = roi_group['rois'][0]['scanfields']['pixelResolutionXY']
    pixelResolution = np.mean(np.divide(FOV, numPX)).round(3).astype(float)

    imageData = []
    roi_x, roi_y = numPX
    for channel in range(totalChannel):
        print(f'Assembling channel {channel + 1} of {totalChannel}...')
        frameTemp = []
        for strip in range(num_rois):
            stripTemp = np.empty((roi_y, roi_x, 1, totalFrame), dtype=np.float32)
            for frame in range(totalFrame):
                frame_data = roi_data[strip].imageData[channel][frame][0]
                stripTemp[:, :, 0, frame] = roi_data[strip].imageData[channel][frame][0]
            corr = return_scan_offset(stripTemp, 1).astype(int)

            stripTemp = fix_scan_phase(np.array(stripTemp), corr, 1)
            val = round(stripTemp.shape[0] * 0.03)
            stripTemp = stripTemp[val:, 7:138, :, :]
            # Concatenate stripTemp to frameTemp along the second axis
            if len(frameTemp) != 0:
                frameTemp = np.concatenate((frameTemp, stripTemp), axis=1)
            else:
                frameTemp = stripTemp
            # Concatenate frameTemp to imageData along the third axis
        if len(imageData) != 0:
            imageData = np.concatenate((imageData, frameTemp), axis=2)
        else:
            imageData = frameTemp
    # Convert imageData to a single precision float type
    imageData = np.array(imageData, dtype=np.float32)

x = 4
