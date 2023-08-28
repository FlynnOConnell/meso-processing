import numpy as np


def fix_scan_phase(data_in, offset, dim):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters:
    -----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.
    dim : int
        Dimension along which to apply the offset.
        1 for vertical (along height/sy), 2 for horizontal (along width/sx).

    Returns:
    --------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """

    sy, sx, sc, sz = data_in.shape
    data_out = None
    if dim == 1:
        if offset > 0:
            data_out = np.zeros((sy, sx + offset, sc, sz))
            data_out[0::2, :sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, offset:offset + sx, :, :] = data_in[1::2, :, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out = np.zeros((sy, sx + offset, sc, sz))  # This initialization is key
            data_out[0::2, offset:offset + sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, :sx, :, :] = data_in[1::2, :, :, :]
        else:
            half_offset = int(offset / 2)
            data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
            data_out[:, half_offset:half_offset + sx, :, :] = data_in

    elif dim == 2:
        data_out = np.zeros(sy, sx, sc, sz)
        if offset > 0:
            data_out[:, 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[offset:(offset + sy), 1::2, :, :] = data_in[:, 1::2, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out[offset:(offset + sy), 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[:, 1::2, :, :] = data_in[:, 1::2, :, :]
        else:
            data_out[int(offset / 2):sy + int(offset / 2), :, :, :] = data_in

    return data_out
