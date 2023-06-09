import settings

import sys
import os

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from mpl_toolkits.mplot3d import Axes3D

from helper import console_general_data_info

INPUT_DATA = sys.argv[1]
VISUAL_DIR = os.environ["VISUAL_DIR"]

def kernel_pca_gif() -> None:
    # Load csv
    data = pd.read_csv(INPUT_DATA,
                       index_col=0)

    console_general_data_info(data)
    # Visualize the clustering with PCA
    pca = KernelPCA(n_components=3, kernel='rbf')
    data = data.head(20000)
    target_columns = data.columns.tolist()
    target_columns.remove("line 1 pump speed")
    target_columns.remove("line 2 pump speed")
    target_columns.remove("PAC pump 1 speed")
    target_columns.remove("PAC pump 2 speed")
    target_columns.remove("year")
    target_columns.remove("time_x")
    target_columns.remove("time_y")
    target_columns.remove("date_x")
    target_columns.remove("date_y")

    principal_components = pca.fit_transform(data[target_columns])

    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    principal_df['label'] = -1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(5):
        category_start = i * 10
        category_end = (i + 1) * 10
        category_label = i

        # Categorize the data
        for index, value in enumerate(data["line 1 pump speed"].values):
            if category_start <= value < category_end:
                principal_df.iloc[[index], principal_df.columns.get_loc('label')] = category_label
        _sub_df = principal_df.loc[principal_df['label'] == category_label]
        # Plot
        ax.scatter(_sub_df['PC1'], _sub_df['PC2'], _sub_df['PC3'], color=colors[i], label=i, alpha=0.3)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Kernal PCA Results')
    plt.savefig(f"{VISUAL_DIR}/kernel_PCA.png", dpi=300)
    # Moving PCA
    PCA_gif_dir = f"{VISUAL_DIR}/kernel_PCA_gif"
    if not os.path.exists(PCA_gif_dir):
        os.mkdir(PCA_gif_dir)
    no_of_frame = 360
    bar = tqdm(total=no_of_frame)
    for i in range(no_of_frame):
        # Set the viewing angle
        ax.view_init(elev=20 + i, azim=30 + i)  # Adjust the elevation and azimuth angle
        plt.savefig(f"{PCA_gif_dir}/PCA_{str(i).zfill(3)}.png", dpi=300)
        bar.update()
    plt.clf()


if __name__ == "__main__":
    kernel_pca_gif()
