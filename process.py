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
from datetime import datetime
import meso_mc
import logging

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    filename=Path().home() / 'data' / "pre-processing.log",
                    level=logging.INFO
                    )

logger = logging.getLogger(__name__)

# Step 1 - Assemble frames into volumes separated into files, one for each plane
def process_files(files, tmp_path):
    num_planes, d1_file, d2_file, t_file = None, None, None, None
    for i, file_path in enumerate(files):
        print(f'Loading file {i + 1} of {len(files)}...')

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

        # Collate the data into a single 4D array
        image_data = util.reorganize(num_px, total_channel, num_rois, total_frame, roi_data)
        full_volume_size = image_data.shape
        d1_file, d2_file, num_planes, t_file = full_volume_size

        # Define the order based on the number of planes
        if num_planes == 30:
            order = [0, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 3, 14, 15, 16, 17, 18, 19,
                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

            order = order[::-1]
        elif num_planes == 15:
            order = [0, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 3, 14]
            order = order[::-1]
        else:
            raise ValueError("Number of planes not recognized.")

        vol = image_data[:, :, order, :]

        # Temporarily save this processed data to disk with via hdf5
        filename = tmp_path / f'{file_path.stem}_temp.h5'

        util.save_to_disk(vol, filename, frame_rate, pixel_resolution, full_volume_size)
        print(f'File {i + 1} processed.')
    print('All files processed.')


def collate_volumes(vol_size, n_files, save_path, idx):
    if not isinstance(vol_size, list):
        # fix this to be more general
        raise ValueError("vol_size must be a list of the form [d1, d2, d3, d4] for unpacking.")

    x, y, _, t = vol_size
    array_in = np.zeros((x, y, t * n_files), dtype=np.float32)
    for file_idx in range(n_files):
        matfilename = temp_names[file_idx]
        with h5py.File(matfilename, 'r') as file:
            vol_data = file['vol'][:, :, plane_idx, :]
            array_in[:, :, file_idx * t:(file_idx + 1) * t] = np.asarray(vol_data).astype(np.float32).reshape(
                (x, y, t))
    array_in -= np.min(array_in)

    # convert w, h, t to t, w, h for CaImAn
    array_in = np.transpose(array_in, (2, 0, 1))

    # save the collated volume to disk
    # Fix this, save_single name logic should NOT be separate from the saving in collate_volumes logic
    util.save_single(array_in, save_path, idx)
    return save_path / f'{idx}.h5'


if __name__ == "__main__":
    import h5py

    number_of_planes = 30

    # dirty setup of paths and filenames
    home = Path.home()
    data_path = Path.home() / "data" / "lbm"

    temp_path = Path(data_path) / "temp"
    fig_path = Path(data_path) / "figs"

    plane_path = Path(data_path) / "planes"
    temp_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    plane_path.mkdir(parents=True, exist_ok=True)

    file_names = sorted(data_path.glob('*.tif'))
    temp_names = sorted(temp_path.glob('*.h5'))
    num_files = len(temp_names)

    # 1 - Assemble frames into volumes separated into files, one for each plane. Likely want to add a check to see if
    #     this has already been done rather than commenting out the code below.
    #
    # process_files(file_names, temp_path)

    # 2 - Motion correct each plane in parallel (or this *should* be in parallel, but isn't as of now)
    vol, volume_rate, pixel_resolution, full_volume_size = util.load_from_disk(temp_names[0])
    multi_file = True

    respath = fig_path / f'raw_shifts'
    respath.mkdir(parents=True, exist_ok=True)
    dview, n_processes = meso_mc.setup_cluster(n_processes=16)
    print(f'Motion correction with {n_processes} processes.')
    try:
        for plane_idx in range(number_of_planes):
            print(f'PROCESSING PLANE {plane_idx + 1} OF {number_of_planes}...')
            name = r'plane_' + str(plane_idx)
            fname = collate_volumes(full_volume_size, num_files, plane_path, name)
            # 2a - Rigid motion correction
            dview = meso_mc.motion_correct(fname, pixel_resolution, respath, dview, plot=False)
    finally:
        meso_mc.teardown_cluster(dview)

