import settings # Get config

from helper import *
from transformer import WaterFormer

import torch
from torch import nn

import sys
from datetime import datetime

try:
    MODEL_NAME = sys.argv[2]
except IndexError:
    print("No model name specified, using current time stamp")
    # Get the current timestamp
    current_time = datetime.now()
    # Format the timestamp as YY-MM-DD-HH-MM
    formatted_time = current_time.strftime("%y-%m-%d-%H-%M-%s")
    MODEL_NAME = f"trans_for_{formatted_time}"

VISUAL_DIR = settings.VISUAL_DIR
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
DEVICE = settings.DEVICE
RAW_DIR = settings.RAW_DIR

# Create working dir
WORKING_DIR = os.path.join(MODEL_DIR, MODEL_NAME)
isResumed = create_folder_if_not_exists(WORKING_DIR)

def main() -> None:
    model: WaterFormer = WaterFormer(
        100,
        14,
        device=DEVICE,
    ).to(DEVICE)

    print(f"Save data to {WORKING_DIR}")
    save_dir = os.path.join(WORKING_DIR, f"{model.name}.onnx")
    example_input = torch.rand((16, 32, 14))
    args = (example_input.to(device=DEVICE), example_input.to(device=DEVICE))
    model.to(device=DEVICE)
    model.eval()
    with torch.autocast(device_type=DEVICE):
        scripted = torch.jit.trace(model, example_inputs=args, check_trace=False) # HACK
    torch.onnx.export(
        model = scripted,
        args = args,
        f = save_dir,
        export_params = True,        # store the trained parameter weights inside the model file
        do_constant_folding = True,  # whether to execute constant folding for optimization
        input_names = ['encoder', 'decoder'],   # the model's input names
        output_names = ['forecast'], # the model's output names
        dynamic_axes = {'encoder' : {0: 'batch_size', 1: 'flatten_encoder_sequence'},    # variable length axes
                        'decoder' : {0: 'batch_size', 1: 'flatten_decoder_sequence'},
                        'forecast': {0: 'batch_size', 1: 'output_sequence'}
                        },
        opset_version = 18,
        )


if __name__ == "__main__":
    main()