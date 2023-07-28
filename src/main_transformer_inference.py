import os
import sys
import onnx
import json
import torch
import settings

import numpy as np
import pandas as pd
import onnxruntime as ort

from transformer import TransformerLossConsolePlotter, WaterFormerDataset, transformer_collate_fn
from helper import to_numeric_and_downcast_data

MODEL_PATH = sys.argv[1]
WORKING_DIR = sys.argv[2]
DEVICE = settings.DEVICE
DEVICE = "cpu"

def load_hyperparameter(working_dir: str) -> dict:
    with open(working_dir, "r", encoding="utf-8") as f:
        result = json.loads(f.read())
    return result

HYPERPARAMETER = load_hyperparameter(os.path.join(WORKING_DIR, "hyperparameter.json"))

def main() -> None:
    # Read data and do split
    data = pd.read_csv(
        os.path.join(WORKING_DIR, "val.csv"),
        low_memory=False,
        index_col=0,
        parse_dates=["timestamp"],
    )

    def dataframe_to_loader(
            data: pd.DataFrame,
            ) -> torch.utils.data.DataLoader:
        # Downcast data
        data = to_numeric_and_downcast_data(data.copy())
        
        # Make sure data is in ascending order by timestamp
        data.sort_values(by=["timestamp"], inplace=True)
        
        # Split data
        src = np.array(data[HYPERPARAMETER["src_columns"]].values)
        tgt = np.array(data[HYPERPARAMETER["tgt_columns"]].values).reshape((-1, 1))

        timestamp = data.reset_index(names="timestamp")["timestamp"].to_numpy(dtype=np.datetime64)
        
        dataset = WaterFormerDataset(
            src,
            tgt,
            timestamp,
            HYPERPARAMETER["knowledge_length"],
            HYPERPARAMETER["pump_dictionary"]["dict_size"],
            device=DEVICE,
        )
        HYPERPARAMETER["spatiotemporal_encoding_size"] = dataset.spatiotemporal_encoding_size
        return dataset
    
    loader = dataframe_to_loader(data)
    
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(
        MODEL_PATH,
        providers = ["CPUExecutionProvider"],
        )

    for i in loader:
        input_args = {
            'encoder': np.array(i[0].unsqueeze(0)),
            'decoder': np.array(i[1].unsqueeze(0)),
        }
        outputs = ort_sess.run(
            output_names=["forecast"], 
            input_feed=input_args,
            run_options=None
            )
        outputs = torch.argmax(torch.softmax(torch.tensor(outputs, device=DEVICE), 2), dim=2)
        print(outputs)

    return

if __name__ == "__main__":
    main()

