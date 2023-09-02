import numpy as np
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.params import CNMFParams
from pathlib import Path
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr, nb_plot_contour
from caiman.motion_correction import MotionCorrect
import caiman as cm
from matplotlib import pyplot as plt


# 1. Setting up the cluster
def setup_cluster():
    locals_dict = locals()
    if 'dview' in locals():
        print('dview found in local variables: killing the cluster')
        cm.stop_server(dview=locals_dict['dview'])
    print('Setting up a cluster')
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    return c, dview, n_processes


# 2. Dataset dependent parameters
def set_dataset_params(fnames: list[str], frate=10, decay_time=0.4, rigid: bool = True):
    pw_rigid = rigid  # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (3, 3)  # size of high pass spatial filtering, used in 1p data
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'  # replicate values along the boundaries

    mc_dict = {
        'fnames': fnames,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    return params.CNMFParams(params_dict=mc_dict)


def perform_motion_correction(savepath, rigid: bool = True, options: CNMFParams = None, local_dview: object = None,
                              border_nan: str = 'copy'):
    # do motion correction rigid
    mc = MotionCorrect(options.get('data', 'fnames'), dview=local_dview, **options.get_group('motion'))
    mc.motion_correct(save_movie=True)
    try:
        fname_mc = mc.fname_tot_els if rigid else mc.fname_tot_rig
    # Propegate our own attribute error
    except AttributeError:
        AttributeError(f'Motion correction does not yet have the correct attribute for'
                       f'correction: Rigid =  {rigid}.. ensure rigid was set in options')
    if rigid:
        border_pixels_cutoff = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                                  np.max(np.abs(mc.y_shifts_els)))).astype(int)
    else:
        border_pixels_cutoff = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
        plt.subplot(1, 2, 1)
        plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2)
        plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.savefig(savepath / 'rigid.png')

    border_pixels_cutoff = 0 if border_nan == 'copy' else border_pixels_cutoff
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=border_pixels_cutoff)
    return fname_new, border_pixels_cutoff


# 4. Load memory-mapped file
def load_memmap_file(fname_new):
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    print('number of images: ', images.shape[0])
    return Yr, dims, T, images


# 5. Setting parameters for CNMF-E
def set_cnmfe_params(options, border_pixels):
    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    merge_thr = .7  # merging threshold, max correlation allowed
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
    min_corr = .8  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    options.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                       'K': K,
                                       'gSig': gSig,
                                       'gSiz': gSiz,
                                       'merge_thr': merge_thr,
                                       'p': p,
                                       'tsub': tsub,
                                       'ssub': ssub,
                                       'rf': rf,
                                       'stride': stride_cnmf,
                                       'only_init': True,  # set it to True to run CNMF-E
                                       'nb': gnb,
                                       'nb_patch': nb_patch,
                                       'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                       'low_rank_background': low_rank_background,
                                       'update_background_components': True,
                                       # sometimes setting to False improve the results
                                       'min_corr': min_corr,
                                       'min_pnr': min_pnr,
                                       'normalize_init': False,  # just leave as is
                                       'center_psf': True,  # leave as is for 1 photon
                                       'ssub_B': ssub_B,
                                       'ring_size_factor': ring_size_factor,
                                       'del_duplicates': True,  # whether to remove duplicates from initialization
                                       'border_pix': border_pixels})  # number of pixels to not consider in the borders)
    print('CNMFE parameters set')
    return options


# 6. Inspect summary images and set parameters
#    Make sure we keep swap_dim=False, our time data is already on the last axis?
def inspect_summary_images(images, gSig):
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=True)
    inspect_correlation_pnr(cn_filter, pnr)
    print(' ***** ')
    print('Corr set')
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
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))
    return cnm


# 9. Plotting
def get_plots(cnm, cn_filter):
    # inspect the results
    # cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
    cnm.estimates.hv_view_components(img=cn_filter, idx=cnm.estimates.idx_components,
                                     denoised_color='red', cmap='gray')
    cnm.estimates.hv_view_components(img=cn_filter, idx=cnm.estimates.idx_components_bad,
                                     denoised_color='red', cmap='gray')
    return None


# Main Execution
if __name__ == "__main__":
    do_mc = False
    ca_path = Path.home() / 'data' / 'ca'
    filename = ca_path / 'Spatial_downsam.tiff'
    savepath = ca_path / 'figs'
    savepath.mkdir(exist_ok=True, parents=True)

    if not filename.exists():
        raise FileNotFoundError(f'File not found: {filename}')
    else:
        filenames = [str(filename)]
        print(f'Files: {filenames}')

    c, dview, n_processes = setup_cluster()
    opts = set_dataset_params(filenames)
    Yr, dims, T, images = None, None, None, None
    if do_mc:
        print('Performing motion correction')
        fname_new, bord_px = perform_motion_correction(savepath, True, opts, dview)
    else:
        print('Skipping motion correction')
        mmap_files = list(Path.home().joinpath('data', 'ca').glob('*.mmap'))
        if len(mmap_files) > 0:
            fname_new = str(mmap_files[0])
            bord_px = 0
            Yr, dims, T, images = load_memmap_file(fname_new)
        else:
            raise FileNotFoundError(f'No .mmap files found in the specified directory')
    opts = set_cnmfe_params(opts, bord_px)
    gSig = opts.get_group('init')['gSig']
    cn_filter, cnr = inspect_summary_images(images, gSig)
    cnm = run_cnmfe(images, n_processes, dview, opts)
    cnm = evaluate_components(cnm, images, dview)
    get_plots(cnm, cn_filter)

    # Stop the cluster
    cm.stop_server(dview=dview)
