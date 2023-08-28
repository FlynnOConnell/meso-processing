import numpy as np
from scipy.signal import correlate, correlation_lags


def return_scan_offset(Iin, dim):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    This function calculates the scan offset correction by analyzing the cross-correlation
    between interleaved lines or columns of the input image. The cross-correlation peak
    determines the amount of offset between the lines or columns, which is then used to
    correct for any misalignment in the imaging process.

    Parameters:
    -----------
    Iin : ndarray
        Input image or volume. It can be 2D, 3D, or 4D. The dimensions represent
        [height, width], [height, width, time], or [height, width, time, channel/plane],
        respectively.
    dim : int
        Dimension along which to compute the scan offset correction.
        1 for vertical (along height), 2 for horizontal (along width).

    Returns:
    --------
    int
        The computed correction value, based on the peak of the cross-correlation.

    Examples:
    ---------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes:
    ------
    This function assumes that the input image contains interleaved lines or columns that
    need to be analyzed for misalignment. The cross-correlation method is sensitive to
    the similarity in pattern between the interleaved lines or columns. Hence, a strong
    and clear peak in the cross-correlation result indicates a good alignment, and the
    corresponding lag value indicates the amount of misalignment.
    """

    if len(Iin.shape) == 3:
        Iin = np.mean(Iin, axis=2)
    elif len(Iin.shape) == 4:
        Iin = np.mean(np.mean(Iin, axis=3), axis=2)

    n = 8

    Iv1 = None
    Iv2 = None
    if dim == 1:
        Iv1 = Iin[::2, :]
        Iv2 = Iin[1::2, :]

        min_len = min(Iv1.shape[0], Iv2.shape[0])
        Iv1 = Iv1[:min_len, :]
        Iv2 = Iv2[:min_len, :]

        buffers = np.zeros((Iv1.shape[0], n))

        Iv1 = np.hstack((buffers, Iv1, buffers))
        Iv2 = np.hstack((buffers, Iv2, buffers))

        Iv1 = Iv1.T.ravel(order='F')
        Iv2 = Iv2.T.ravel(order='F')

    elif dim == 2:
        Iv1 = Iin[:, ::2]
        Iv2 = Iin[:, 1::2]

        min_len = min(Iv1.shape[1], Iv2.shape[1])
        Iv1 = Iv1[:, :min_len]
        Iv2 = Iv2[:, :min_len]

        buffers = np.zeros((n, Iv1.shape[1]))

        Iv1 = np.vstack((buffers, Iv1, buffers))
        Iv2 = np.vstack((buffers, Iv2, buffers))

        Iv1 = Iv1.ravel()
        Iv2 = Iv2.ravel()

    # Zero-center and clip negative values to zero
    Iv1 = Iv1 - np.mean(Iv1)
    Iv1[Iv1 < 0] = 0

    Iv2 = Iv2 - np.mean(Iv2)
    Iv2[Iv2 < 0] = 0

    Iv1 = Iv1[:, np.newaxis]
    Iv2 = Iv2[:, np.newaxis]

    r_full = correlate(Iv1[:, 0], Iv2[:, 0], mode='full', method='auto')
    unbiased_scale = len(Iv1) - np.abs(np.arange(-len(Iv1) + 1, len(Iv1)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - n
    upper_bound = mid_point + n + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-n, n + 1)

    # Step 3: Find the correction value
    correction_index = np.argmax(r)
    return lags[correction_index]