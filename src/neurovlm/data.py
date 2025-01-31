"""Fetch required data."""

import os
from typing import Optional
import os
import requests

URL = "https://github.com/neurovlm/neurovlm_data/raw/refs/heads/main/data"

def fetch_data(url: Optional[str]=URL, overwrite: Optional[bool]=False) -> str:
    """Fetch data from a url.

    Parameters
    ----------
    url : string, optional, default: URL
        URL to data.
    overwrite : bool, optional, default: False
        Overwrite existing files.

    Returns
    -------
    save_dir : string
        Path to saved data.
    """
    files = ["mask.npz", "publications.parquet", "coordinates.parquet"]

    save_dir = os.path.dirname(os.path.realpath(__file__))
    if "neurovlm_data" not in os.listdir(save_dir):
        os.mkdir(save_dir + "/neurovlm_data")
    elif not overwrite:
        return save_dir + "/neurovlm_data"

    save_dir = save_dir + "/neurovlm_data"

    for f in files:

        # File url
        file_url = f"{url}/{f}"

        # Fetch
        response = requests.get(file_url)

        if response.status_code == 200:
            # Save file
            with open(f"{save_dir}/{f}", "wb") as file:
                file.write(response.content)
        else:
            print(f"Error fetching '{f}': {response.status_code}")

    return save_dir