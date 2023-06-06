import settings # Read the config

import sys
import os

from tqdm import tqdm
from termcolor import colored

import pandas as pd
import numpy as np
import seaborn as sns
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from mpl_toolkits.mplot3d import Axes3D

from helper import console_general_data_info

INPUT_DATA = sys.argv[1]
VISUAL_DIR = os.environ["VISUAL_DIR"]
PROCESSED_DIR = os.environ["DATA_DIR"]

print(f"Read from {INPUT_DATA}")

def visual_data_distribution(saving_dir: str, data: pd.DataFrame) -> None:
    # Data distribution analysie
    data_distribution_dir = f"{VISUAL_DIR}/{saving_dir}"
    if not os.path.exists(data_distribution_dir):
        os.mkdir(data_distribution_dir)
    bar = tqdm(total=data.shape[1])
    for column in data:
        try:
            file_name = f"{column[0]}.png"
            plt.hist(data[column].values, bins=30)
            plt.title(f"Data distribution figure of {column[1]}")
            plt.savefig(f"{data_distribution_dir}/{file_name}", dpi=300)
            plt.clf()
        except ValueError:
            print(f"{column[0]} contains invalid values. Skipped")
        bar.update()
    return

def main() -> None:
    # Load csv
    data = pd.read_csv(INPUT_DATA, 
                       header=[0,1], 
                       index_col=0, 
                       parse_dates=[1])
    
    console_general_data_info(data)

    # Time data
    data[("f31", "date")] = [pd.to_datetime(d[0]).date() for d in data.loc[:, "dt"].values.astype(np.datetime64)]
    data[("f32", "time")] = [pd.to_datetime(d[0]).time() for d in data.loc[:, "dt"].values.astype(np.datetime64)]
    data = data.drop(columns=["dt"])
    
    # Convert abs time to rel time
    start_date = data.iloc[0, data.columns.get_loc(("f31", "date"))]
    print(f"The relative date starts at {start_date}")
    data[("f31", "date")] = pd.to_datetime(data[("f31", "date")]) - pd.to_datetime(start_date)
    data[('f31', 'date')] = data[('f31', 'date')].dt.days.astype(int)
    start_time = pd.to_datetime('00:00:00', format= '%H:%M:%S')
    data[('f32', 'time')] = pd.to_datetime(data[('f32', 'time')], format= '%H:%M:%S') - start_time
    data[('f32', 'time')] = data[('f32', 'time')].dt.total_seconds().astype(int)
    
    console_general_data_info(data)
    
    # Remove unreasonable information
    ## Drop all empty columns
    mask = data.apply(lambda col: all(col.isin([0, np.nan])), axis=0)
    data = data.loc[:, ~mask]

    # Raw data distribution
    visual_data_distribution("data_distribution_raw", data)

    # Remove unreasonable information
    ## f1 remove maxima number
    data = data[data[('f1', '进水流量m3/h')] != 10000]
    ## Drop f14, f19
    data = data.drop(columns=["f14", "f19"])
    ## f12-f13 round to zeros
    data.loc[data[('f12', '生化池1加药泵频率')] < 0.2, ('f12', '生化池1加药泵频率')] = 0
    data.loc[data[('f13', '生化池2泵频率')] < 0.2, ('f13', '生化池2泵频率')] = 0
    ## f16-f18 round to zeros
    data.loc[data[('f16', 'pac泵1频率')] < 0.2, ('f16', 'pac泵1频率')] = 0
    data.loc[data[('f17', 'pac泵2频率')] < 0.2, ('f17', 'pac泵2频率')] = 0
    data.loc[data[('f18', 'pac泵3频率')] < 0.2, ('f18', 'pac泵3频率')] = 0
    ## Data zero and NaN value fix
    ## Remove the row with row being all zeros
    ## Print out base information for the data
    ## Drop enitrely empty and unusable data
    mask = data.iloc[:, 1:].apply(lambda row: all(row.isin([0, np.nan])), axis=1)
    data = data[~mask]

    ## See results
    print("\nAfter dropping:")
    console_general_data_info(data)

    #data.to_csv(f"{OUTPUT_DIR}/pured.csv")

    # calculate the variance
    header = ["Feature",
              "Average",
              "Variance",
              "Coeff Var (%)",
              "Max",
              "Min"
              ]
    infos = []
    for column, cnt in zip(data, range(data.shape[1])):
        if column in [('f31', 'date'), ('f32', 'time')]:
            continue
        info = [column, np.average(data[column].values),
                np.var(data[column].values),
                np.var(data[column].values)/np.mean(data[column].values)*100,
                np.max(data[column].values),
                np.min(data[column].values)
                ]
        infos.append(info)
    print(tb.tabulate(infos, header))
    ## Notes on the basic column information
    ## For most features, the coeff of variance is really big,
    ## The features that have relatively small coeff var can have missing values filled in with average number.
    ## Features with very big coeff var should be dropped if there are missing values.
    ## Features that is perferablely filled in with average when missing:
    ## - f5
    ## - f7
    ## - f8
    ## - f10
    ## - f11
    ## Features that is OK to be zero
    ## - f12
    ## - f13
    ## - f16
    ## - f17
    ## - f18
    # Calculate how many rows have missing numbers
    all_zeros = (data == 0).astype(int).sum(axis=0)
    unfixable_zeros = (data[["f1", "f2", "f3", "f4", "f6", "f9"]] == 0).astype(int).sum(axis=0).sum(axis=0)
    unfixable_zeros += (data[["f1", "f2", "f3", "f4", "f6", "f9"]] == np.nan).astype(int).sum(axis=0).sum(axis=0)
    fixable_zeros = (data[["f5", "f7", "f8", "f10", "f11"]] == 0).astype(int).sum(axis=0).sum(axis=0)
    print(f"Rows with fixable zeros: {fixable_zeros}")
    print(f"Rows with unfixable zeros: {unfixable_zeros}")
    
    # How many cannot
    ## Drop unfixable
    data = data[data[('f1', '进水流量m3/h')] != 0]
    data = data[data[('f2', '进水cod mg/l')] != 0]
    data = data[data[('f3', '进水氨氮')] != 0]
    data = data[data[('f4', '进水tn')] != 0]
    data = data[data[('f6', '出水cod')] != 0]
    data = data[data[('f9', '出水tp')] != 0]
    data = data[~data[('f9', '出水tp')].isnull()]
    ## Fix fixable
    data.loc[data[('f5', '进水tp')] == 0, ('f5', '进水tp')] = np.average(data[data[('f5', '进水tp')] != 0].loc[:, ('f5', '进水tp')].values)
    data.loc[data[('f7', '出水氨氮')] == 0, ('f7', '出水氨氮')] = np.average(data[data[('f7', '出水氨氮')] != 0].loc[:, ('f7', '出水氨氮')].values)
    data.loc[data[('f8', '出水tn')] == 0, ('f8', '出水tn')] = np.average(data[data[('f8', '出水tn')] != 0].loc[:, ('f8', '出水tn')].values)
    data.loc[data[('f10', '生化池1硝酸盐氮')] == 0, ('f10', '生化池1硝酸盐氮')] = np.average(data[data[('f10', '生化池1硝酸盐氮')] != 0].loc[:, ('f10', '生化池1硝酸盐氮')].values)
    data.loc[data[('f11', '生化池2硝酸盐氮')] == 0, ('f11', '生化池2硝酸盐氮')] = np.average(data[data[('f11', '生化池2硝酸盐氮')] != 0].loc[:, ('f11', '生化池2硝酸盐氮')].values)

    # Data distribution analysis
    visual_data_distribution("data_distribution", data)

    # Data relation and corelationship anaylsis
    correlation_matrix = data.corr(numeric_only=True)
    mask = correlation_matrix.apply(lambda row: all(row.isin([0, np.nan])), axis=1)
    correlation_matrix = correlation_matrix[~mask]
    mask = correlation_matrix.apply(lambda col: all(col.isin([0, np.nan])), axis=0)
    correlation_matrix = correlation_matrix.loc[:, ~mask]
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(f"{VISUAL_DIR}/correlation.png", dpi=300)
    plt.clf()
    ## Notes about the correlationship
    ## As we could see in the figure, variables with strong corelation (>0.4) are
    ## - f3, f4
    ## - f4, f5
    ## - f1, f18
    ## - f6, f19
    ## - f9, f11
    ## - f10, f11
    ## - f12, f13
    ## - f18, f19
    ## As a result, we decide to drop feature:
    ## f4, f18, f19, f12
   
    # data =  data.drop(columns = ["f4", "f18", "f12"])
    console_general_data_info(data)

    print(colored("Saving data", "green"))
    data.to_csv(f"{PROCESSED_DIR}/processed.csv")

    # Data normalization and scaling
    print("\nNormalization and scaling data")
    for column in data:
        if column in [('f16','pac泵1频率'),('f17','pac泵2频率'),('f18','pac泵3频率'),('f31', 'date'), ('f32', 'time')]:
            continue
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.reshape(np.array(data[column]), (-1, 1)))
        # transformer = PowerTransformer(method='yeo-johnson')
        # mapped_data = transformer.fit_transform(scaled_data)
        # data[column] = mapped_data
        data[column] = scaled_data
        # print(f"{column[0]} contains invalid values. Skipped")
    console_general_data_info(data)
    
    print(colored("Saving data", "green"))
    data.to_csv(f"{PROCESSED_DIR}/processed_norm.csv")

    # Data distribution analysis
    visual_data_distribution("data_distribution_normed", data)
    return

if __name__ == "__main__":
    main()
