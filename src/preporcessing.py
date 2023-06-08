import settings # Read the config

import os
import sys
import math
import datetime

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
DATA_DIR = os.environ["DATA_DIR"]

print(f"Read from {INPUT_DATA}")

def visual_data_distribution(saving_dir: str, data: pd.DataFrame) -> None:
    # Data distribution analysis
    data_distribution_dir = f"{VISUAL_DIR}/{saving_dir}"
    if not os.path.exists(data_distribution_dir):
        os.mkdir(data_distribution_dir)
    bar = tqdm(total=data.shape[1])
    for index, column in enumerate(data):
        try:
            file_name = f"f{index}: {column}.png"
            plt.hist(data[column].values, bins=30)
            plt.title(f"Data distribution figure of {column}")
            plt.savefig(f"{data_distribution_dir}/{file_name}", dpi=300)
            plt.clf()
        except ValueError:
            print(f"{column} contains invalid values. Skipped")
        bar.update()
    return

def visualize_variance(data: pd.DataFrame) -> None:
    # calculate the variance
    target_columns = data.columns.tolist()
    target_columns.remove("timestamp")
    header = ["Feature",
              "Average",
              "Variance",
              "Coeff Var (%)",
              "Max",
              "Min"
              ]
    infos = []
    for column in target_columns:
        info = [column, np.average(data[column].values),
                np.var(data[column].values),
                np.var(data[column].values)/np.mean(data[column].values)*100,
                np.max(data[column].values),
                np.min(data[column].values)
                ]
        infos.append(info)
    print(tb.tabulate(infos, header))
    print()

def visualize_count_zeros(data: pd.DataFrame) -> None:
    # Count zero values in each features
    print(colored("Counting zeros in each feature", "green"))
    zero_cnts = []
    for col in data:
        zero_cnts.append((col, (data[col] == 0).sum()))
    print(tb.tabulate(zero_cnts, ["Col", "Zeros"]))
    print()

def visualize_correlation(data: pd.DataFrame) -> None:
    # Data relation and correlationship anaylsis
    correlation_matrix = data.corr(numeric_only=True)
    mask = correlation_matrix.apply(lambda row: all(row.isin([0, np.nan])), axis=1)
    correlation_matrix = correlation_matrix[~mask]
    mask = correlation_matrix.apply(lambda col: all(col.isin([0, np.nan])), axis=0)
    correlation_matrix = correlation_matrix.loc[:, ~mask]
    plt.figure(figsize=(10, 8))
    plt.title('Correlation Matrix')
    sns.set(font_scale=0.5)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.savefig(f"{VISUAL_DIR}/correlation.png", dpi=300)
    plt.clf()
    sns.set(font_scale=1)
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

def drop_zeros(data: pd.DataFrame) -> pd.DataFrame:
    ## Data zero and NaN value fix
    ## Remove the row with row being all zeros
    ## Print out base information for the data
    ## Drop enitrely empty and unusable data
    mask = data.iloc[:, 1:].apply(lambda row: all(row.isin([0, np.nan])), axis=1)
    data = data[~mask]
    
    # Fill NaN with 0
    data = data.fillna(0)
    
    # Remove unreasonable information
    ## f1 remove maxima number
    data = data[data[('inlet flow')] != 10000]
    ## f12-f13 round to zeros
    data.loc[data["line 1 pump speed"] < 0.2, 'line 1 pump speed'] = 0
    data.loc[data["line 2 pump speed"] < 0.2, 'line 2 pump speed'] = 0
    ## f16-f18 round to zeros
    data.loc[data["PAC pump 1 speed"] < 0.2, "PAC pump 1 speed"] = 0
    data.loc[data["PAC pump 2 speed"] < 0.2, "PAC pump 2 speed"] = 0
    
    # Drop record
    ## Zeros to NaN
    target_columns = data.columns.tolist()
    target_columns.remove("timestamp")
    target_columns.remove("line 1 pump speed")
    target_columns.remove("line 2 pump speed")
    target_columns.remove("PAC pump 1 speed")
    target_columns.remove("PAC pump 2 speed")
    print(colored(f"Removing empty data from \n{target_columns}", "green"))
    data[target_columns] = data[target_columns].replace(0, np.nan)
    data = data.dropna(thresh=0.7*len(target_columns), subset=target_columns)
    data = data.fillna(0)

    ## See results
    print("\nAfter dropping:")
    console_general_data_info(data)

    return data

def fill_zeros(data: pd.DataFrame) -> pd.DataFrame:
    target_columns = data.columns.tolist()
    target_columns.remove("timestamp")
    target_columns.remove("line 1 pump speed")
    target_columns.remove("line 2 pump speed")
    target_columns.remove("PAC pump 1 speed")
    target_columns.remove("PAC pump 2 speed")
    
    for column in target_columns:
        # Find all the zero sequence
        zero_sequences = []
        cnt = 0
        start_index = 0
        isCounting = False
        for index, row in enumerate(data[column].values):
            if row == 0:
                if isCounting:
                    cnt += 1
                else:
                    cnt = 1
                    start_index = index
                isCounting = True
            else:
                if isCounting:
                    zero_sequences.append((start_index, cnt))
                    isCounting = False
        print(colored(f"{column} has {len(zero_sequences)} zero sequence,", "green"))
        print(tb.tabulate(zero_sequences, ["Starting index", "Length"]))
        # Drop zero sequence that is longer than 50
        # Fill in the zero sequence using polyfit

        for starting_index, length in zero_sequences:
            important_columns = ["timestamp", column]
            target_length = 10
            # Find non-zero sequence before
            for_length = 0
            for i in range(target_length):
                testee = data.iloc[starting_index - i - 1, data.columns.get_loc(column)]
                if testee != 0:
                    for_length += 1
                else:
                    break
            for_sequence = data.iloc[starting_index - 1 - for_length: starting_index].copy()
            for_sequence = for_sequence[important_columns]
            # Find non-zero sequence after
            aft_length = 0
            for i in range(target_length):
                testee = data.iloc[starting_index + length + i, data.columns.get_loc(column)]
                if testee != 0:
                    aft_length += 1
                else:
                    break
            aft_sequence = data.iloc[starting_index + length: starting_index + length + aft_length].copy()
            aft_sequence = aft_sequence[important_columns]
            # The unknwon sequence
            unknown_sequence = data.iloc[starting_index: starting_index + length].copy()
            unknown_sequence = unknown_sequence.loc[:, important_columns]
            # Combine the sequence
            known_sequence = pd.concat([for_sequence, aft_sequence])
            # Convert the timestamp
            time_format = "%Y-%m-%d %I:%M:%S"
            starting_time = pd.to_datetime(known_sequence.iloc[0].copy(deep=True)["timestamp"], format=time_format)
            known_sequence.loc[:, "timestamp"] = pd.to_datetime(known_sequence["timestamp"], format=time_format)
            known_sequence.loc[:, "time"] = [int((t - starting_time).total_seconds()) for t in known_sequence["timestamp"]]
            unknown_sequence.loc[:, "timestamp"] = pd.to_datetime(unknown_sequence["timestamp"], format=time_format)
            unknown_sequence.loc[:, "time"] = [int((t - starting_time).total_seconds()) for t in unknown_sequence["timestamp"]]
            # polyfit the sequence
            p = np.polyfit(known_sequence["time"].values, known_sequence[column].values, 1)
            # Estimate the unknown
            guess = np.polyval(p, unknown_sequence.loc[:, "time"])
            # print(f"Guess {guess}")
            # Put the sequence back
            data.iloc[starting_index:starting_index + length, data.columns.get_loc(column)] = guess
    
    return data

def create_more_data(data: pd.DataFrame) -> pd.DataFrame:
    # Creating more data
    # Time data
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["year"] = [pd.to_datetime(d).year for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
    data["date"] = [pd.to_datetime(d).date() for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
    data["time"] = [pd.to_datetime(d).time() for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
    data = data.drop(columns=["timestamp"])

    # Convert abs time to rel time
    ## Year
    start_year = data.iloc[0, data.columns.get_loc("year")]
    print(f"The relative year start at {start_year}")
    data["year"] = data["year"] - start_year
    ## Date
    start_date = data.iloc[0, data.columns.get_loc("date")]
    print(f"The relative date starts at {start_date}")
    data["date"] = pd.to_datetime(data["date"]) - pd.to_datetime(start_date)
    data['date'] = data['date'].dt.days.astype(int)
    data["date_x"] = [math.cos(d / 365 * math.pi) for d in data.loc[:, "date"].values]
    data["date_y"] = [math.sin(d / 365 * math.pi) for d in data.loc[:, "date"].values]
    data = data.drop(columns=["date"])
    ## Time
    start_time = pd.to_datetime('00:00:00', format= '%H:%M:%S')
    data['time'] = pd.to_datetime(data['time'], format= '%H:%M:%S') - start_time
    data['time'] = data['time'].dt.total_seconds().astype(int)
    data["time_x"] = [math.cos(t / 86400 * math.pi) for t in data.loc[:, "time"].values]
    data["time_y"] = [math.cos(t / 86400 * math.pi) for t in data.loc[:, "time"].values]
    data = data.drop(columns=["time"])
    """
    #  Create non-ammonia nitrogen data
    data["inlet non-ammonia nitrogen"] = data["inlet total nitrogen"] - data["inlet ammonia nitrogen"]
    data = data.drop(columns=["inlet total nitrogen"])
    data["outlet non-ammonia nitrogen"] = data["outlet total nitrogen"] - data["outlet ammonia nitrogen"]
    data = data.drop(columns=["outlet total nitrogen"])
    # Rearrange columns
    cols = data.columns.tolist()
    cols = cols[13:18] + cols[:13] + cols[18:]
    cols = cols[:8] + cols[-2:-1] + cols[8:11] + cols[-1:] + cols[11:-2]
    data = data[cols]
    """
    # Rearrange columns
    cols = data.columns.tolist()
    cols = cols[15:20] + cols[:15] + cols[20:]
    data = data[cols]

    return data

def normalization_and_scaling(data: pd.DataFrame) -> pd.DataFrame:
    # Data normalization and scaling
    print(colored("Normalization and scaling data", "green"))
    print()
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

    for column in target_columns:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.reshape(np.array(data[column]), (-1, 1)))
        # transformer = PowerTransformer(method='yeo-johnson')
        # mapped_data = transformer.fit_transform(scaled_data)
        # data[column] = mapped_data
        data[column] = scaled_data
    return data


def main() -> None:
    # Load csv
    data = pd.read_csv(INPUT_DATA, 
                       index_col=0, 
                       parse_dates=[1],
                       date_format="%Y/%m/%d %I:%M %p"
                       )
    
    console_general_data_info(data)
    # Raw data distribution
    visual_data_distribution("data_distribution_raw", data.drop(columns=["timestamp"]))
    
    # Drop zeros
    print(colored("Data property before processing:", "green"))
    visualize_variance(data)
    data = drop_zeros(data)
    print(colored("Data property after processing:", "green"))
    visualize_variance(data)
    visual_data_distribution("data_distribution_dropped", data.drop(columns=["timestamp"]))
    
    """
    Notes on the basic column information
    For most features, the coeff of variance is really big,
    The features that have relatively small coeff var can have missing values filled in with average number.
    Features with very big coeff var should be dropped if there are missing values.
    Features that is perferablely filled in with average when missing:
    - inlet phosphorus  
    - f5
    - f7
    - f8
    - f10
    - f11
    Features that is OK to be zero
    - line 1 pump speed
    - line 2 pump speed
    - PAC pump 1 speed
    - PAC pump 2 speed
    """

    # Fill zeros
    visualize_count_zeros(data)
    data = fill_zeros(data)
    print(colored("Double check if the fill zero process is successful.", "green"))
    visualize_count_zeros(data)
    visualize_variance(data)

    # Add more data
    console_general_data_info(data)
    data = create_more_data(data)
    visualize_correlation(data)
    console_general_data_info(data)
    visual_data_distribution("data_distribution_filled", data)

    # Saving data
    print(colored("Saving data", "green"))
    data.to_csv(f"{DATA_DIR}/processed.csv")

    # Normalization and scaling data
    data = normalization_and_scaling(data)
    console_general_data_info(data)
    visual_data_distribution("data_distribution_normed", data)
    
    # Saving data
    print(colored("Saving data", "green"))
    data.to_csv(f"{DATA_DIR}/processed_norm.csv")
    return

if __name__ == "__main__":
    main()
