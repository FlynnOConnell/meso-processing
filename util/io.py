import tables
from pathlib import Path


def save_to_disk(vol, filename, volume_rate, pixel_resolution, full_volume_size):
    """Save the volume to disk in a HDF5 file. Pytables is used for optimized chunking and compression."""

    # Make sure the filename is a Path object to avoid operating system errors
    savepath = Path(filename) if isinstance(filename, str) else filename

    with tables.open_file(savepath.with_suffix('.h5'), mode='w') as f:
        atom = tables.Atom.from_dtype(vol.dtype)
        filters = tables.Filters(complevel=5, complib='blosc')

        #  Pay attention to not specify chunkshape here,
        #  otherwise the data will be stored in non-contiguously
        ds = f.create_carray(f.root, 'vol', atom, vol.shape, filters=filters)
        ds[:] = vol

        # Store the metadata in a separate group
        group = f.create_group("/", 'metadata')

        f.create_array(group, 'volume_rate', volume_rate)
        f.create_array(group, 'pixel_resolution', pixel_resolution)
        f.create_array(group, 'full_volume_size', full_volume_size)


# We may want this for other storage formats in the future
def determine_chunk_size(shape, target_chunk_size):
    """Determine a suitable chunk size based on the dataset shape and a target chunk size."""
    chunk_size = []
    for dim_size, target_dim_chunk_size in zip(shape, target_chunk_size):
        # Find a divisor of dim_size that is close to the target dimension chunk size
        divisors = [i for i in range(1, dim_size + 1) if dim_size % i == 0]
        closest_divisor = min(divisors, key=lambda x: abs(x - target_dim_chunk_size))
        chunk_size.append(closest_divisor)
    return tuple(chunk_size)
