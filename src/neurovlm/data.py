"""Fetch required data."""

import os
from pathlib import Path
from typing import Optional
import requests

URL = "https://github.com/neurovlm/neurovlm_data/raw/refs/heads/main/data"

def fetch_data(
    url: Optional[str]=URL,
    files: Optional[list]=None,
    overwrite: Optional[bool]=False
) -> str:
    """Fetch data from a url.

    Parameters
    ----------
    url : string, optional, default: URL
        URL to data.
    files : list, optional, default: None
        Files to download. None defaults to all files.
    overwrite : bool, optional, default: False
        Overwrite existing files.

    Returns
    -------
    save_dir : string
        Path to saved data.
    """
    if files is None:
        files = ["mask.npz", "publications.parquet", "coordinates.parquet",
                 "decoder_half.pt", "aligner_half.pt"]

    save_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    if "neurovlm_data" not in os.listdir(save_dir):
        os.mkdir(save_dir / "neurovlm_data")
    elif not overwrite:
        return save_dir / "neurovlm_data"

    save_dir = save_dir / "neurovlm_data"

    for f in files:

        # File url
        file_url = f"{url}/{f}"

        # Fetch
        response = requests.get(file_url)

        if response.status_code == 200:
            # Save file
            with open(save_dir / f, "wb") as file:
                file.write(response.content)
        else:
            print(f"Error fetching '{f}': {response.status_code}")

    return Path(save_dir)