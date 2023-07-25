import settings # Read the config

import os
import sys
import json

from tqdm import tqdm
from typing import Tuple
from termcolor import colored, cprint
from gauss_rank_transformation import GaussRankScaler

import pandas as pd
import numpy as np
import seaborn as sns
import tabulate as tb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from helper import console_general_data_info, create_folder_if_not_exists

INPUT_DATA = sys.argv[1]
print(colored(f"Read from {INPUT_DATA}", "black", "on_green"))
print()

VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
RAW_DIR = settings.RAW_DIR

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
    _visual_data_distribution("dis_raw", data)
    _visualize_data_trend("trend_raw", data)

    # Read national standards
    def load_national_standards() -> dict:
        """Load national standard from data/GB

        Returns:
            dict: scaling factors in dict format 
                result:
                    chemical: standard\n
                    chemical: standard\n
                    chemical: standard
        """
        data = pd.read_csv(
            os.path.join(
            RAW_DIR, "GB18918-2002.csv"
            ),
            index_col=0,
        )
        result = {}
        for index, row in data.iterrows():
            result[row[0]] = row[1]
        return result
    national_standards = load_national_standards()

    # Drop zeros
    print(colored("Data property before processing:", "green"))
    _visualize_variance(data, data.columns.tolist())

    def mark_unreasonable_data(data: pd.DataFrame) -> pd.DataFrame:
        # Remove unreasonable information
        ## Remove maxima number, reset it to zero,
        ## so that it will be process in the following steps
        data['inlet flow'] = data['inlet flow'].replace(10000, np.NaN)
        ## Remove close to zero pump speed
        for column in Y_COLUMNS:
            data.loc[data[column] < 0.2, column] = 0

        # Mark record
        print(colored(f"Removing empty data from \n{X_COLUMNS}", "green"))
        data[X_COLUMNS] = data[X_COLUMNS].replace(0, np.nan)

        ## Drop entirely empty and unusable data
        mask = data.apply(lambda row: all(row.isin([0, np.nan])), axis=1)
        data = data[~mask]

        ## Drop wired data
        mask = data[Y_COLUMNS].apply(lambda row: all(row.isin([0, np.nan])), axis=1)
        data = data[~mask]

        # Fill pump 1 with pump 2 data
        data["line 1 pump speed"].iloc[(data["line 1 pump speed"] == 0) & (data["line 2 pump speed"] != 0)] = data["line 2 pump speed"].loc[(data["line 1 pump speed"] == 0) & (data["line 2 pump speed"] != 0)]

        # Remove non sense pump speed
        mask = data[["line 1 pump speed", "line 2 pump speed"]].apply(lambda row: all(row.isin([0, np.nan])), axis=1)
        data = data[~mask]
        return data
    
    data = mark_unreasonable_data(data)
    console_general_data_info(data)
    _visual_data_distribution("dis_reasonable", data)

    # Gauss rank transformation
    def gauss_rank_transformation(data: pd.DataFrame, national_standards: dict) -> Tuple[pd.DataFrame, dict]:
        # Data transformation and scaling
        print(colored("Transformation and scaling data", "green"))
        print()
        name_mapping = {
            "outlet COD": "COD",
            "outlet ammonia nitrogen": "ammonia nitrogen",
            "outlet total nitrogen": "total nitrogen",
            "outlet phosphorus": "total phosphorus",
        }
        for column in X_COLUMNS:
            transformer = GaussRankScaler()
            x = data[column].to_numpy(na_value=np.nan).reshape(-1, 1)
            data[column] = transformer.fit_transform(x)
            if column in name_mapping:
                # Convert the raw national standards to the scaled ones
                national_standards[name_mapping[column]] = transformer.transform(
                    np.array(
                    national_standards[name_mapping[column]]
                    ).reshape(1, 1)
                    )[0, 0]

        return data, national_standards
    
    data, scaled_national_standards = gauss_rank_transformation(data, national_standards)
    cprint(f"Scaled national standards: {national_standards}\n", "green")
    _visual_data_distribution("dis_gauss", data)
    console_general_data_info(data)

    def one_hot_label(data: pd.DataFrame, dict_size: int) -> pd.DataFrame:
        """
        Create one hot label for the data
        """
        unique_values = data[TGT_COLUMNS].unique()
        unique_values = np.sort(unique_values)
        cprint(f"Unique value:", "green")
        print(unique_values)
        pump_speed_upper_bound = unique_values[-1] # Upper bound will always be 100%, it is probably 50
        pump_speed_lower_bound = unique_values[0] # Lower bound is probably 0

        word_dictionary = {
            "overload": pump_speed_upper_bound,
            "start": pump_speed_lower_bound,
            "full": unique_values[-1],
            "dict_size": dict_size,
        }
        discrete_tgt_column = f"{TGT_COLUMNS} discrete"
        data[discrete_tgt_column] = np.nan
        def _judge(x: float, start: float, stop: float) -> bool:
            if x >= start and x < stop:
                return True
            else:
                return False
        possible_words = np.linspace(unique_values[0], unique_values[-2] + 0.000001, num=dict_size - 1)
        for word, (lower, upper) in enumerate(zip(possible_words[:-1], possible_words[1:]), start=1):
            mask = data[TGT_COLUMNS].apply(lambda x: _judge(x, lower, upper))
            data[discrete_tgt_column].iloc[mask] = word
        
        data[discrete_tgt_column].iloc[data[TGT_COLUMNS] == pump_speed_upper_bound] = 0

        return data, word_dictionary
    
    data, word_dictionary = one_hot_label(data, 100)
    
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
    
    # Store the dictionary
    dictionary_path = os.path.join(DATA_DIR, "pump_dictionary.json")
    with open(dictionary_path, "w", encoding="utf-8") as f:
        json.dump(word_dictionary, f, indent=2)

    # Store the national standards
    scaled_national_standards_path = os.path.join(DATA_DIR, "scaled_national_standards.json")
    with open(scaled_national_standards_path, "w", encoding="utf-8") as f:
        json.dump(scaled_national_standards, f, indent=2)
    return

if __name__ == "__main__":
    main()