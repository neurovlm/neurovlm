"""Extract coordinates from publication tables, specifically biorxiv."""
import os
import contextlib
import numpy as np
import pandas as pd
import camelot

def extract_coords_from_table(file_path: str, pg: int | str, columns: list[int]) -> np.ndarray:
    """Extract MNI coordinates from tables in publication PDFs.

    Parameters
    ----------
    file_path : str
        Path to .pdf file.
    pg : str | int
        Page in pdf to extract table from.
    columns : list[int]
        Which columns contain x, y, z coordinates.
s
    Returns
    -------
    np.ndarray
        Matrix of x, y, z MNI coordinates, extract from table.

    Examples
    --------
    >>> file_path = "/path/to/publication.pdf"
    >>> coords = extract_coords_from_table_(file_path=file_path, pg=23, columns=[2, 3, 4])
    """
    # Read in table(s) from page
    with open(os.devnull, 'w') as fnull: # suppress stdout and stderr
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            tables = camelot.read_pdf(
                file_path,
                pages=str(pg),
                flavor="stream",
                column_tol=10000,      # parse data as a single column
                suppress_stdout=True,
            )

    # Expand columns using \n delimiter
    coords = tables[0].df[0].str.split("\n", expand=True)

    # Mask out non-numeric rows (e.g. header may overflow into second row)
    row_mask = pd.to_numeric(coords.iloc[:, columns[0]], errors="coerce").notna()

    return coords[row_mask].iloc[:, columns].values.astype(int)