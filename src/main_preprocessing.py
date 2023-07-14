import settings # Read the config

import os
import sys
import math
import json

from tqdm import tqdm
from datetime import timedelta
from termcolor import colored, cprint

import pandas as pd
import numpy as np
import seaborn as sns
import tabulate as tb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split

from helper import console_general_data_info, create_folder_if_not_exists, new_remove

INPUT_DATA = sys.argv[1]
print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print()

VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR

META = {
    "random_state": 42,
}

RAW_COLUMNS = [
    "inlet flow",
    "inlet COD",
    "inlet ammonia nitrogen",
    "inlet total nitrogen",
    "inlet phosphorus",
    "outlet COD",
    "outlet ammonia nitrogen",
    "outlet total nitrogen",
    "outlet phosphorus",
    "line 1 nitrate nitrogen",
    "line 2 nitrate nitrogen",
    "line 1 pump speed",
    "line 2 pump speed",
    "PAC pump 1 speed",
    "PAC pump 2 speed",
]

TIME_COLUMNS = [
    "year",
    "date_x",
    "date_y",
    "time_x",
    "time_y",
]

X_COLUMNS = RAW_COLUMNS[:-4]
Y_COLUMNS = RAW_COLUMNS[-4:]

TGT_COLUMNS = RAW_COLUMNS[-4]

# Helper

def _visual_data_distribution(saving_dir: str, data: pd.DataFrame) -> None:
    # Data distribution analysis
    data_distribution_dir = f"{VISUAL_DIR}/{saving_dir}"
    create_folder_if_not_exists(data_distribution_dir)
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

def _visualize_variance(
        data: pd.DataFrame,
        target_columns: list,
        ) -> None:
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

def _visualize_count_zeros(data: pd.DataFrame) -> None:
    # Count zero values in each features
    print(colored("Counting zeros in each feature", "green"))
    zero_cnts = []
    for col in data:
        zero_cnts.append((col, (data[col] == 0).sum()))
    print(tb.tabulate(zero_cnts, ["Col", "Zeros"]))
    print()

def _visualize_correlation(data: pd.DataFrame) -> None:
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

def _visualize_data_trend(name: str, data: pd.DataFrame) -> None:
    working_dir = os.path.join(VISUAL_DIR, name)
    create_folder_if_not_exists(working_dir)

    for i in data:
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the data
        ax.plot(data[i], linewidth=2)
            
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Data')
        ax.set_title('Prediction Trend Plot')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Customize the tick labels
        ax.tick_params(axis='x', rotation=45)

        # Add a background color
        ax.set_facecolor('#F2F2F2')

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        plt.title(" ".join(i.split("_")))

        # Show the plot
        plt.savefig(
            os.path.join(
                working_dir, 
                f"{i}_trend.png"
                ), 
            dpi = 400,
            format = "png")
        plt.clf()
        plt.close()
    return

def create_pump_data_off_zone(data: pd.DataFrame) -> pd.DataFrame:
    """This function deal with the problem of pump function being very discrete when off

    Args:
        data (pd.DataFrame): All the data

    Returns:
        pd.DataFrame: All the data
    """
    cprint("Remove 0 from covariance, making data more smooth", "green")
    target_columns = [
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ]
    # Because 0 is far from other values, it is better to make it closer to other values
    for i in target_columns:
        # Find the second smallest value in the column
        second_smallest = np.partition(data[i].unique(), 1)[1] * 1000

        # Find the smallest value in the column
        smallest = data[i].min()

        # Report
        cprint(f"Find the smallest value in {i}, it is {smallest}", "green")
        cprint(f"Find the second smallest value in {i}, it is {second_smallest/1000}", "green")

        # Convert all occurrences of the smallest value to the floor rounding of the second smallest value
        data.loc[data[i] == smallest, i] = math.floor(second_smallest) / 1000
    return data

def one_hot_label(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create one hot label for the data
    """
    target_columns = [
        "line 1 pump speed",
        "line 2 pump speed",
        "PAC pump 1 speed",
        "PAC pump 2 speed",
    ]
    skip_columns = []
    for i in target_columns:
        # Temporarily store data
        temp = data[i].copy()

        # Destructive procedure for data
        data[i] *= 2
        data[i] = data[i].round(-1)
        data[i] /= 10
        data[i] = data[i].astype(np.int8)

        # Generate one-hot labels for each value in the range
        for j in range(0, 11): # The pump speed is expected to be in 0-50
            skip_columns.append(f"{i} {j}")
            data[f"{i} {j}"] = (data[i] == j).astype(np.uint8)
        # Recover data
        data[i] = temp
    
    return data, skip_columns

# Main

def main() -> None:
    # Load csv
    cprint("Loading csv.", color="black", on_color="on_cyan", attrs=["blink"])
    data = pd.read_csv(INPUT_DATA, 
                       index_col=0, 
                       parse_dates=[1],
                       date_format="%Y/%m/%d %I:%M %p"
                       )
    console_general_data_info(data)
    # Set index ad timestamp
    data = data.set_index("timestamp")
    ## Raw data distribution
    _visual_data_distribution("distribution_raw", data)
    _visualize_data_trend("trend_raw", data)

    # Drop zeros
    print(colored("Data property before processing:", "green"))
    _visualize_variance(data, data.columns.tolist())

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
        ## Remove maxima number, reset it to zero,
        ## so that it will be process in the following steps
        data['inlet flow'] = data['inlet flow'].replace(10000, 0)
        ## Remove close to zero pump speed
        for column in Y_COLUMNS:
            data.loc[data[column] < 0.2, column] = 0 # TODO: Remove 0s in a way that make the data more consistent

        # Drop record
        print(colored(f"Removing empty data from \n{X_COLUMNS}", "green"))
        data[X_COLUMNS] = data[X_COLUMNS].replace(0, np.nan)
        data = data.dropna(thresh=0.7*len(X_COLUMNS), subset=X_COLUMNS)
        data = data.fillna(0)

        ## See results
        print("\nAfter dropping:")
        console_general_data_info(data)

        return data
    
    data = drop_zeros(data)
    print(colored("Data property after processing:", "green"))
    _visualize_variance(data, data.columns.tolist())

    # Fill zeros
    cprint("Filling zeros.", color="black", on_color="on_cyan", attrs=["blink"])
    _visualize_count_zeros(data)

    def fill_zeros(data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index(names="timestamp")
        for column in X_COLUMNS:
            # Find all the zero sequence
            ## zero_sequence: [(start_index: int, length of the zero sequence: int)]
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
            for start_index, cnt in zero_sequences:

                pass
            # Put the time stamp back
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
                # The unknown sequence
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
                parameters = np.polyfit(known_sequence["time"].values, known_sequence[column].values, 1)
                # Estimate the unknown
                guess = np.polyval(parameters, unknown_sequence.loc[:, "time"])
                # print(f"Guess {guess}")
                # Put the sequence back
                data.iloc[starting_index:starting_index + length, data.columns.get_loc(column)] = guess
        data = data.set_index("timestamp")
        return data
    
    data = fill_zeros(data)
    print(colored("Double check if the fill zero process is successful.", "green"))
    _visualize_count_zeros(data)
    _visualize_variance(data, data.columns.tolist())

    # Add more data
    cprint("Adding more data.", color="black", on_color="on_cyan", attrs=["blink"])
    console_general_data_info(data)
    
    def create_more_data(data: pd.DataFrame) -> pd.DataFrame:
        data = data.reset_index(names="timestamp")
        # Creating more data
        # Time data
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["year"] = [pd.to_datetime(d).year for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
        data["date"] = [pd.to_datetime(d).date() for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
        data["time"] = [pd.to_datetime(d).time() for d in data.loc[:, "timestamp"].values.astype(np.datetime64)]
        # data = data.drop(columns=["timestamp"])
        data = data.set_index("timestamp")

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

        # Rearrange columns
        cols = data.columns.tolist()
        cols = cols[15:20] + cols[:15] + cols[20:]
        data = data[cols]
        
        return data
    
    data = create_more_data(data)
    _visualize_correlation(data)
    console_general_data_info(data)

    # Normalization and scaling data
    cprint("Normalizing and scaling data.", color="black", on_color="on_cyan", attrs=["blink"])

    def normalization_and_scaling(data: pd.DataFrame, skip_columns: list=[]) -> pd.DataFrame:
        # Data normalization and scaling
        print(colored("Normalization and scaling data", "green"))
        print()
        target_columns = data.columns.tolist()
        for i in skip_columns:
            target_columns.remove(i)

        scaling_factors = {}
        for column in target_columns:
            # Standardization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(np.reshape(np.array(data[column]), (-1, 1)))
            average = scaler.mean_[0]
            stddev = scaler.scale_[0]
            parameters = scaler.get_params()
            # Storing
            meta = [
                average,
                stddev,
                parameters,
            ]
            scaling_factors[column] = meta
            data[column] = scaled_data

        return data, scaling_factors

    def transformation_and_scaling(data: pd.DataFrame, skip_columns: list=[]) -> pd.DataFrame:
        # Data transformation and scaling
        print(colored("Transformation and scaling data", "green"))
        print()
        target_columns = data.columns.tolist()
        for i in skip_columns:
            target_columns.remove(i)

        scaling_factors = {}
        for column in target_columns:
            # Normalization
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            transformed_data = transformer.fit_transform(np.reshape(np.array(data[column]), (-1, 1)))
            n_quantiles_ = transformer.n_quantiles_
            quantiles_ = transformer.quantiles_.tolist()
            references_ = transformer.references_.tolist()
            n_features_in_ = transformer.n_features_in_
            parameters = transformer.get_params()
            # Storing
            meta = [
                    n_quantiles_, 
                    quantiles_,
                    references_,
                    n_features_in_,
                    parameters,
                    ]
            scaling_factors[column] = meta
            data[column] = transformed_data

        return data, scaling_factors
    
    _ = Y_COLUMNS.copy()
    _.extend(TIME_COLUMNS)
    data, x_scaling_factors = normalization_and_scaling(data, skip_columns=_)
    _ = X_COLUMNS.copy()
    _.extend(TIME_COLUMNS)
    data, y_scaling_factors = transformation_and_scaling(data, skip_columns=_)
    console_general_data_info(data)
    _visualize_variance(data, data.columns.tolist())
    _visual_data_distribution("distribution_normed", data)
    _visualize_data_trend("trend_final", data)

    # Saving data
    cprint("Saving data.", color="black", on_color="on_cyan", attrs=["blink"])
    def save_data_csv(path: str, 
                  name: str, 
                  data: pd.DataFrame,
                  split: bool = True,
                  **kwarg
                  ) -> None:
        """Save data to csv

        Args:
            path (str): directory for saving csv
            name (str): file name, not including ".csv"
            data (pd.DataFrame): data needs to be saved
            split (bool, optional): Wether do train_test_split. Defaults to True.
        """
        print(colored("Saving data", "green"))
        if not split:
            # Save
            save_dir = os.path.join(path, f"{name}.csv")
            data.to_csv(save_dir)
        else:
            # Create folder
            working_dir = os.path.join(path, name)
            create_folder_if_not_exists(working_dir)
            # Split
            train_data, test_data = train_test_split(data, **kwarg)
            train_data, val_data = train_test_split(train_data, **kwarg)
            # Save
            save_dir_train = os.path.join(working_dir, f"{name}_train.csv")
            train_data.to_csv(save_dir_train)
            save_dir_val = os.path.join(working_dir, f"{name}_val.csv")
            val_data.to_csv(save_dir_val)
            save_dir_test = os.path.join(working_dir, f"{name}_test.csv")
            test_data.to_csv(save_dir_test)
        print()
        return

    save_data_csv(DATA_DIR,
                  "processed", 
                  data, 
                  split=False, 
                  )
    x_scaling_factors_path = os.path.join(DATA_DIR, "x_scaling_factors.json")
    with open(x_scaling_factors_path, "w", encoding="utf-8") as f:
        json.dump(x_scaling_factors, f, indent=2)
    y_scaling_factors_path = os.path.join(DATA_DIR, "y_scaling_factors.json")
    with open(y_scaling_factors_path, "w", encoding="utf-8") as f:
        json.dump(y_scaling_factors, f, indent=2)

    # Split data
    def split_and_save_data(data, name: str) -> None:
        # Split data
        cprint("Splitting data.", color="black", on_color="on_cyan", attrs=["blink"])
        """
        The subprocess split data into multiple dataframe based on the time delta
        """
        data.sort_values(by=["timestamp"], inplace=True)
        # Get the time delta
        data = data.reset_index(names="timestamp")
        
        data["timedelta"] = data["timestamp"].diff()
        # Convert the time delta column to minutes
        data['time_delta_minutes'] = data['timedelta'].astype('timedelta64[s]')
    
        """
        Typically normal timedelta is 10 minutes
        However, it is possible to have 9 minutes, 11 minutes
        I will manually filter those out, as they are not too far from 10 minutes
        And they create too much segments
        """
    
        normal_timedelta = np.timedelta64(10, 'm')
        min_9_timedelta = np.timedelta64(9, 'm')
        min_11_timedelta = np.timedelta64(11, 'm')
        # Filter the DataFrame for time deltas equal to 10 minutes (600s)
        data = data[
            (data['time_delta_minutes'] == normal_timedelta)
            | (data['time_delta_minutes'] == min_9_timedelta)
            | (data['time_delta_minutes'] == min_11_timedelta)
            ]
    
        data = data.drop(columns=["timedelta", "time_delta_minutes"])
    
        console_general_data_info(data)
    
        # Split data into multiple section
        data["timedelta"] = data["timestamp"].diff()
        ## Find the indices where the time difference is not 10 minutes
        split_indices = data.index[
            (data["timedelta"] != pd.Timedelta(minutes=10))
            & (data["timedelta"] != pd.Timedelta(minutes=9))
            & (data["timedelta"] != pd.Timedelta(minutes=11))
            ]
        ## Split the DataFrame into multiple DataFrames based on the split indices
        data = data.drop(columns=["timedelta"])
        dfs = np.split(data, split_indices)

        # Remove excessive information
        data_segments = dfs

        # Save split data
        working_dir = os.path.join(DATA_DIR, name)
        create_folder_if_not_exists(working_dir)
        bar = tqdm(total=len(data_segments), desc=colored("Saving data", color="green", attrs=["blink"]))
        for index, data in enumerate(data_segments):
            data = data.reset_index(drop=True)
            data = data.set_index("timestamp")
            data.to_csv(
                os.path.join(
                    working_dir, f"seg_{index}.csv"
                )
            )
            bar.update()
        bar.close()
        return
    
    split_and_save_data(data, "segmented_data")
    return

if __name__ == "__main__":
    main()
