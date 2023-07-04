import settings

import sys
import os

from tqdm import tqdm

import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D
from termcolor import cprint

from helper import console_general_data_info, create_folder_if_not_exists

INPUT_DIR = sys.argv[1] # Input should be the segmented data
VISUAL_DIR = settings.VISUAL_DIR

Y_COLUMNS = [
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]
def generate_skip_columns():
    """
    Skip the one-hot label columns
    """
    skip_columns = []
    for column in Y_COLUMNS:
        for i in range(11):
            skip_columns.append(f"{column} {i}")
    return skip_columns
SKIP_COLUMNS = generate_skip_columns()

def _find_biggest_csv_file(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    if not csv_files:
        return None

    biggest_file = max(csv_files, key=lambda file: os.path.getsize(os.path.join(directory, file)))
    return os.path.join(directory, biggest_file)

def kernel_pca_gif() -> None:
    csv_file = _find_biggest_csv_file(INPUT_DIR)
    # Load csv
    data = pd.read_csv(csv_file,
                       index_col=0)

    # Create dir
    console_general_data_info(data)
    working_dir = os.path.join(VISUAL_DIR, "dft")
    create_folder_if_not_exists(working_dir)

    # Define target to plot
    target_columns = data.columns.tolist()
    target_columns.remove("year")
    target_columns.remove("time_x")
    target_columns.remove("time_y")
    target_columns.remove("date_x")
    target_columns.remove("date_y")
    for i in SKIP_COLUMNS:
        target_columns.remove(i)

    # Plot
    with tqdm(target_columns) as bar:
        for target_column in bar:
            # DFT analysis
            dft_result = sp.fft.fft(data[target_column].values)
            n = dft_result.size
            time_step = 0.1
            frequencies = sp.fft.fftfreq(n, d=time_step)
            
            # Plot
            dft_df = pd.DataFrame({'Frequency': frequencies, 'Amplitude': np.abs(dft_result)})
            plt.plot(dft_df['Frequency'], dft_df['Amplitude'])
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.title('Discrete Fourier Transform')
            plt.grid(True)
            plt.savefig(f"{working_dir}/{target_column}.png", dpi=300)
            plt.cla()
    cprint("Finish plotting!", "green")

if __name__ == "__main__":
    kernel_pca_gif()
