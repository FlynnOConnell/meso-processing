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
import numpy as np
import util
import ScanImageTiffReader


def process_files(data_path):
    data_path = Path(data_path)
    file_names = sorted(data_path.glob('*.tif'))

    for i, file_path in enumerate(file_names):
        print(f'Loading file {i + 1} of {len(file_names)}...')

        with ScanImageTiffReader.ScanImageTiffReader(str(file_path)) as reader:
            metadata_str = reader.metadata()
            data = reader.data()

        # Deal with the non-json serializable string returned from the ScanImageTiffReader
        metadata_kv, metadata_json = util.parse(metadata_str)

        num_channels = 30
        num_volumes = data.shape[0] // num_channels
        num_rois = len(metadata_json["RoiGroups"]["imagingRoiGroup"]["rois"])
        data = data.reshape(num_volumes, num_channels, *data.shape[1:]).transpose(
            2, 3, 1, 0
        )

        roi_data, roi_group = util.get_mroi_data_from_tiff(
            metadata_json, metadata_kv, data, num_channels, num_volumes, num_rois
        )

        #  Parse metadata from ScanImage file
        total_frame = len(roi_data[0].imageData[0])  # num_frames, same as num_volumes
        total_channel = len(roi_data[0].imageData)  # num_channels, same as num_planes
        frame_rate = metadata_kv["SI.hRoiManager.scanVolumeRate"]
        size_xy = roi_group["rois"][0]["scanfields"]["sizeXY"]
        fov = np.round(157.5 * np.array(size_xy), 0).astype(int)
        num_px = roi_group["rois"][0]["scanfields"]["pixelResolutionXY"]
        pixel_resolution = np.mean(np.divide(fov, num_px)).round(3).astype(float)

        # Collate the data into a single 4D numpy array
        image_data = util.reorganize(num_px, total_channel, num_rois, total_frame, roi_data)
        full_volume_size = image_data.shape
        d1_file, d2_file, number_of_planes, t_file = full_volume_size

        # Define the order based on the number of planes
        if number_of_planes == 30:
            order = [0, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 3, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

            order = order[::-1]
        elif number_of_planes == 15:
            order = [0, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 3, 14]
            order = order[::-1]
        else:
            raise ValueError("Number of planes not recognized.")

        vol = image_data[:, :, order, :]

        # Temporarily save this processed data to disk with via hdf5
        temp_path = data_path / 'temp'
        temp_path.mkdir(exist_ok=True)
        filename = temp_path / f'{file_path.stem}_temp.h5'

        util.save_to_disk(vol, filename, frame_rate, pixel_resolution, full_volume_size)
        print(f'File {i + 1} processed.')
    print('All files processed.')


if __name__ == "__main__":
    home = Path.home()
    datapath = home / "data" / "lbm"
    process_files(datapath)

    x = 4
