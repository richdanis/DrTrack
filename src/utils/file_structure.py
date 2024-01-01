import os

def create_dir(path: str):
    """
    Create a directory at the specified path if it doesn't already exist.

    Parameters
    ----------
    path : str
        The path of the directory to be created.

    Returns
    -------
    None
        This function does not return anything.

    """
    if not os.path.exists(path):
        os.makedirs(path)
