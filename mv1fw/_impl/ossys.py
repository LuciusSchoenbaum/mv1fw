



import os




# helper
def create_dir(base_dir, stem = None):
    """
    Each stem, including the base, is created
    if it does not exist. The stem can be a list.
    If base_dir is None current working directory is used.

    Cf. mkdir -p (p for parent)
    Cf. mkdirs() in Python Standard Library

    Arguments:

        base_dir (string):
            A base directory, assumed to exist at time of call.
        stem (None, string, or list of optional strings)
            A "stem" or valid directory name,
            or else a list of stems, not assumed to exist at time of call.
            If any stem is None it is passed over silently.

    :meta private:
    """
    if isinstance(base_dir, tuple):
        raise ValueError(f"This API was removed! See docs/source code for usage.")
    if not isinstance(stem, list):
        stems = [stem] if stem is not None else []
    else:
        stems = stem
    out = base_dir
    if out is None:
        out = os.getcwd()
    # todo this precaution should not be needed, review
    if not os.path.exists(out):
        os.mkdir(out)
    for stm in stems:
        if stm is not None:
            out = os.path.join(out, stm)
            if not os.path.exists(out):
                # Note: this will fail if it's required to
                # create more than one directory ("stem");
                # that's why this code is here in the first place.
                os.mkdir(out)
    return out






