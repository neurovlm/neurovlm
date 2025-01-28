"""Fetch required data."""

import os
from typing import Optional
import os
import requests

URL = "https://github.com/neurovlm/neurovlm_data/raw/refs/heads/main/data"

def fetch_data(url:Optional[str]=URL):
    """Fetch data from a url.

    Parameters
    ----------
    url : string, optional, default: URL
        URL to data.

    Returns
    -------
    save_dir : string
        Path to saved data.
    """
    files = ["mask.npz"]

    save_dir = os.path.dirname(os.path.realpath(__file__))
    if "neurovlm_data" not in os.listdir(save_dir):
        os.mkdir(save_dir + "/neurovlm_data")
    save_dir = save_dir + "/neurovlm_data"

    for f in files:

        # Construct the full URL for the file
        file_url = f"{url}/{f}"

        # Fetch the file
        response = requests.get(file_url)

        if response.status_code == 200:
            # Save the file to the 'data' directory
            with open(f"{save_dir}/{f}", "wb") as file:
                file.write(response.content)  # Write binary data to file
        else:
            print(f"Error fetching '{f}': {response.status_code}")

    return save_dir