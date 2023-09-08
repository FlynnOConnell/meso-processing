import numpy as np
from util.scan import return_scan_offset
from util.scan import fix_scan_phase


def reorganize(num_px, num_channels, num_rois, num_frames, roi_data):
    imageData = []
    roi_x, roi_y = num_px
    for channel in range(num_channels):
        print(f'Assembling channel {channel + 1} of {num_channels}...')
        frameTemp = []
        for strip in range(num_rois):
            stripTemp = np.empty((roi_y, roi_x, 1, num_frames), dtype=np.float32)
            for frame in range(num_frames):
                frame_data = roi_data[strip].imageData[channel][frame][0]
                stripTemp[:, :, 0, frame] = roi_data[strip].imageData[channel][frame][0]
            corr = return_scan_offset(stripTemp, 1).astype(int)

            stripTemp = fix_scan_phase(np.array(stripTemp), corr, 1)
            val = round(stripTemp.shape[0] * 0.03)
            stripTemp = stripTemp[val - 1:, 6:138, :, :]
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
    return imageData.astype(int)
