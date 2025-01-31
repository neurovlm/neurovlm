"""Report loop progress."""
from typing import Callable

def select_tqdm() -> Callable:
    """Import tqdm based on environment.

    Returns
    -------
    tqdm : tqdm.tqdm or tqdm.notebook.tqdm

    """
    from IPython import get_ipython
    ipython = get_ipython()

    if ipython is None:
        # Script
        from tqdm import tqdm
    elif "IPKernelApp" in ipython.config:
        # Jupyter
        from tqdm.notebook import tqdm
    else:
        # IPython
        from tqdm import tqdm

    return tqdm
