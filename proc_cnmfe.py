from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.params import CNMFParams
from pathlib import Path
from caiman.utils.visualization import (
    inspect_correlation_pnr,
    nb_inspect_correlation_pnr,
    nb_plot_contour,
)
from caiman.motion_correction import MotionCorrect
import caiman as cm
from matplotlib import pyplot as plt
import numpy as np


# 1. Setting up the cluster
def setup_cluster():
    locals_dict = locals()
    if "dview" in locals():
        print("dview found in local variables: killing the cluster")
        cm.stop_server(dview=locals_dict["dview"])
    print("Setting up a cluster")
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local", n_processes=None, single_thread=False
    )
    return c, dview, n_processes


# 2. Dataset dependent parameters
def set_dataset_params(
    fnames: list[str], frate=10, decay_time=0.4, rigid: bool = False
):
    pw_rigid = rigid  # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (3, 3)  # size of high pass spatial filtering, used in 1p data
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (
        48,
        48,
    )  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = (
        3  # maximum deviation allowed for patch with respect to rigid shifts
    )
    border_nan = "copy"  # replicate values along the boundaries

    mc_dict = {
        "fnames": fnames,
        "fr": frate,
        "decay_time": decay_time,
        "pw_rigid": pw_rigid,
        "max_shifts": max_shifts,
        "gSig_filt": gSig_filt,
        "strides": strides,
        "overlaps": overlaps,
        "max_deviation_rigid": max_deviation_rigid,
        "border_nan": border_nan,
    }
    return params.CNMFParams(params_dict=mc_dict)


def perform_motion_correction(
    rigid: bool = True,
    options: CNMFParams = None,
    local_dview: object = None,
    border_nan: str = "copy",
):
    # do motion correction rigid
    mc = MotionCorrect(
        options.get("data", "fnames"), dview=local_dview, **options.get_group("motion")
    )
    mc.motion_correct(save_movie=True)

    return mc


# 4. Load memory-mapped file
def load_memmap_file(fname_new):
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order="F")
    print("number of images: ", images.shape[0])
    return Yr, dims, T, images


# 5. Setting parameters for CNMF-E
def set_cnmfe_params(options, border_pixels):
    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    merge_thr = 0.7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20  # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2  # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1  # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0  # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0  # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = 0.8  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    options.change_params(
        params_dict={
            "method_init": "corr_pnr",  # use this for 1 photon
            "K": K,
            "gSig": gSig,
            "gSiz": gSiz,
            "merge_thr": merge_thr,
            "p": p,
            "tsub": tsub,
            "ssub": ssub,
            "rf": rf,
            "stride": stride_cnmf,
            "only_init": True,  # set it to True to run CNMF-E
            "nb": gnb,
            "nb_patch": nb_patch,
            "method_deconvolution": "oasis",  # could use 'cvxpy' alternatively
            "low_rank_background": low_rank_background,
            "update_background_components": True,
            # sometimes setting to False improve the results
            "min_corr": min_corr,
            "min_pnr": min_pnr,
            "normalize_init": False,  # just leave as is
            "center_psf": True,  # leave as is for 1 photon
            "ssub_B": ssub_B,
            "ring_size_factor": ring_size_factor,
            "del_duplicates": True,  # whether to remove duplicates from initialization
            "border_pix": border_pixels,
        }
    )  # number of pixels to not consider in the borders)
    print("CNMFE parameters set")
    return options


# 6. Inspect summary images and set parameters
#    Make sure we keep swap_dim=False, our time data is already on the last axis?
def inspect_summary_images(images, gSig):
    cn_filter, pnr = cm.summary_images.correlation_pnr(
        images[::1], gSig=gSig[0], swap_dim=True
    )
    inspect_correlation_pnr(cn_filter, pnr)
    print(" ***** ")
    print("Corr set")
    return cn_filter, pnr


# 7. Run CNMF-E algorithm
def run_cnmfe(images, n_processes, dview, opts):
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)
    return cnm


# 8. Component Evaluation
def evaluate_components(cnm, images, dview):
    min_SNR = 3  # adaptive way to set threshold on the transient size
    r_values_min = 0.85  # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set(
        "quality", {"min_SNR": min_SNR, "rval_thr": r_values_min, "use_cnn": False}
    )
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(" ***** ")
    print("Number of total components: ", len(cnm.estimates.C))
    print("Number of accepted components: ", len(cnm.estimates.idx_components))
    return cnm


# 9. Plotting
def get_plots(cnm, cn_filter):
    # inspect the results
    # cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
    cnm.estimates.hv_view_components(
        img=cn_filter,
        idx=cnm.estimates.idx_components,
        denoised_color="red",
        cmap="viridis",
    )
    cnm.estimates.hv_view_components(
        img=cn_filter,
        idx=cnm.estimates.idx_components_bad,
        denoised_color="red",
        cmap="viridis",
    )
    return None


def plot_shifts(shifts, title="Shifts"):
    # %% Plot rigid shifts
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mc.shifts_rig)
    plt.xlabel("frames")
    plt.ylabel("pixels")
    plt.title(f"{title}")


# Plot the first frame of the video
def plot_first_frame(images):
    plt.imshow(images[0], cmap="viridis")
    plt.title("First Frame")
    plt.colorbar()
    plt.show()


# Plot mean image across time
def plot_mean_image(images):
    mean_img = np.mean(images, axis=0)
    plt.imshow(mean_img, cmap="viridis")
    plt.title("Mean Image")
    plt.colorbar()
    plt.show()


# Plot time series for a specific pixel
def plot_time_series(Yr, dims, x, y):
    time_series = Yr[y * dims[1] + x, :]
    plt.plot(time_series)
    plt.title(f"Time Series at pixel ({x}, {y})")
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.show()


# Plot histogram of intensities for first frame
def plot_histogram_first_frame(images):
    plt.hist(images[0].ravel(), bins=100)
    plt.title("Histogram of Intensities in First Frame")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()


# Show a montage of first 10 frames
def plot_montage(images):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap="viridis")
        ax.set_title(f"Frame {i+1}")
    plt.show()


def find_large_motion_jumps(shifts, threshold=10):
    """
    Find points where large jumps in motion occur.

    Parameters:
    - shifts: array of shifts, each element corresponding to a frame.
    - threshold: the minimum difference in shifts to consider as a large jump.

    Returns:
    - jump_frames: list of frame indices where large jumps occur.
    """
    # Calculate the differences between each frame's shift and the previous frame's
    shift_diffs = np.diff(shifts, axis=0)

    # Calculate the magnitude of shift differences
    shift_magnitudes = np.linalg.norm(shift_diffs, axis=1)

    # Find the frames where the magnitude of the shift difference exceeds the threshold
    jump_frames = (
        np.where(shift_magnitudes > threshold)[0] + 1
    )  # +1 to account for the diff operation

    return jump_frames


def inspect_correlation_pnr(correlation_image_pnr, pnr_image):
    """
    inspect correlation and pnr images to infer the min_corr, min_pnr

    Args:
        correlation_image_pnr: ndarray
            correlation image created with caiman.summary_images.correlation_pnr

        pnr_image: ndarray
            peak-to-noise image created with caiman.summary_images.correlation_pnr
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot correlation image
    ax1 = axs[0]
    im_cn = ax1.imshow(correlation_image_pnr, cmap='jet', aspect='auto')
    ax1.set_title('Correlation Image')
    fig.colorbar(im_cn, ax=ax1)

    # Plot PNR image
    ax2 = axs[1]
    im_pnr = ax2.imshow(pnr_image, cmap='jet', aspect='auto')
    ax2.set_title('PNR')
    fig.colorbar(im_pnr, ax=ax2)

    plt.tight_layout()
    plt.show()


# Main Execution
if __name__ == "__main__":
    do_mc = False
    ca_path = Path.home() / "data" / "ca"
    filename = ca_path / "Spatial_downsam.tiff"
    savepath = ca_path / "figs"
    savepath.mkdir(exist_ok=True, parents=True)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    else:
        filenames = [str(filename)]
        print(f"Files: {filenames}")

    c, dview, n_processes = setup_cluster()
    opts = set_dataset_params(filenames)
    Yr, dims, T, images = None, None, None, None
    print("Performing motion correction")
    motion = False
    if motion:
        mc = perform_motion_correction(True, opts, dview)
        fname_new = mc.fname_tot_els if motion else mc.fname_tot_rig
        plot_shifts(mc.shifts_rig, title="Rigid shifts")
    else:
        fname_new = str(Path.home().joinpath("data", "ca", "Spatial_downsam.tiff"))
    # plot shifts

    print("Skipping motion correction")
    mmap_files = list(Path.home().joinpath("data", "ca").glob("*.mmap"))
    if len(mmap_files) > 0:
        fname_new = str(mmap_files[0])
        bord_px = 0
        Yr, dims, T, images = load_memmap_file(fname_new)
    else:
        raise FileNotFoundError(f"No .mmap files found in the specified directory")

    # inspect the file
    plot_first_frame(images)
    plot_mean_image(images)
    plot_time_series(
        Yr, dims, 50, 50
    )  # Change 50, 50 to the x, y coordinates you are interested in

    opts = set_cnmfe_params(opts, bord_px)
    gSig = opts.get_group("init")["gSig"]
    cn_filter, cnr = inspect_summary_images(images, gSig)
    cnm = run_cnmfe(images, n_processes, dview, opts)
    cnm = evaluate_components(cnm, images, dview)

    # Stop the cluster
    cm.stop_server(dview=dview)
