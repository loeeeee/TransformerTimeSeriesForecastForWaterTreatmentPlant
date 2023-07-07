import json
import os

ROOT_DIR = ""
DATA_DIR = ""
CONFIG_DIR = ""
VISUAL_DIR = ""
SRC_DIR = ""
MODEL_DIR = ""
RANGE_DIR = ""

def main() -> None:
    print("[INFO] Settings: Start initializing.")

    DEFAULT_CONFIG_DIR = (
        os.path.dirname(os.path.realpath(__file__)) + "/../config/"
    )  # just in case some one what to modify this

    check_and_create_dir(DEFAULT_CONFIG_DIR)
    vars = read_from_config_file(DEFAULT_CONFIG_DIR)
    default_env_vars = generate_default_values(DEFAULT_CONFIG_DIR)

    if not vars:  # No file was found, use default one
        vars = default_env_vars
    else:  # If found, use the found one, and check them
        vars = env_list_integrity_check(default_env_vars, vars)

    save_to_config_file(DEFAULT_CONFIG_DIR, vars)  # Save change to config
    apply_env_from_list(vars)

    print("[INFO] Settings: Finish initializing.")
    return vars


# subprocess

def generate_default_values(DEFAULT_CONFIG_DIR):
    # generate default value and write to file
    vars = {}

    vars["ROOT_DIR"] = (
        os.path.dirname(os.path.realpath(__file__)) + "/../"
    )  # Root means the repo root
    vars["DATA_DIR"] = os.path.join(vars["ROOT_DIR"], "data/")
    vars["CONFIG_DIR"] = DEFAULT_CONFIG_DIR
    vars["VISUAL_DIR"] = os.path.join(vars["ROOT_DIR"], "visual/")
    vars["SRC_DIR"] = os.path.join(vars["ROOT_DIR"], "src/")
    vars["MODEL_DIR"] = os.path.join(vars["ROOT_DIR"], "model/")
    vars["RANGE_DIR"] = os.path.join(vars["ROOT_DIR"], "range/")
    # env_vars["ROOT_DIR"] =

    return vars


def read_from_config_file(CONFIG_DIR):
    # read values from files and apply
    print("[INFO] Settings: Reading from file.")
    env_vars = {}
    try:
        with open(
            os.path.join(CONFIG_DIR, "config.json"), "r", encoding="utf8"
        ) as the_file:
            env_vars = json.loads(the_file.read())
    except FileNotFoundError:
        print("[WARNING] Settings: Config file not found. Generating a default one.")
    return env_vars


def apply_env_from_list(env_var_list):
    print("[INFO] Settings: Apply environment variable.")
    # Create needed dir
    check_and_create_dir(env_var_list["DATA_DIR"])
    check_and_create_dir(env_var_list["CONFIG_DIR"])
    check_and_create_dir(env_var_list["MODEL_DIR"])
    check_and_create_dir(env_var_list["VISUAL_DIR"])
    check_and_create_dir(env_var_list["RANGE_DIR"])
    # Loading actual value
    for key, value in env_var_list.items():
        print("[INFO] Settings: %s - %s" % (key, value))
    return


def env_list_integrity_check(source, sus):
    # check the integrity of the list. Add the missing variables with default values.
    print("[INFO] Settings: Performing integrity check.")
    checked = sus.copy()
    missing = source.copy()
    for source_var in source:
        for sus_var in sus:
            if source_var == sus_var:
                missing.pop(source_var)

    if missing:
        for missing_var in missing:
            checked[missing_var] = missing[missing_var]
            print(
                "[WARNING] Settings: Missing environment variable, %s, using default value, %s, will save to file."
                % (missing_var, missing[missing_var])
            )

    return checked


def save_to_config_file(DIR, CONTENT):
    with open(os.path.join(DIR, "config.json"), "w", encoding="utf8") as the_file:
        json.dump(CONTENT, the_file, indent=2)
        print("[INFO] Settings: New config file generated.")
    return


# helper
def check_and_create_dir(DIR):
    isExist = os.path.exists(DIR)
    if not isExist:
        print(
            "[WARNING] Settings: Directory %s does not exist. It is created now." % DIR
        )
        try:
            os.mkdir(DIR)
        except FileNotFoundError:
            print(
                "[ERROR] Settings: Intermediate folders in Dir %s does not exist. Please check settings and config."
                % DIR
            )
            raise FileNotFoundError
    else:
        print(
            "[INFO] Settings: Folder already exists in Dir %s, skipping creation." % DIR
        )

    return


isAlreadyExecuted = False
if __name__ == "__main__":
    main()
else:
    if not isAlreadyExecuted:
        vars = main()
        ROOT_DIR    = vars["ROOT_DIR"]
        DATA_DIR    = vars["DATA_DIR"]
        CONFIG_DIR  = vars["CONFIG_DIR"]
        VISUAL_DIR  = vars["VISUAL_DIR"]
        SRC_DIR     = vars["SRC_DIR"]
        MODEL_DIR   = vars["MODEL_DIR"]
        RANGE_DIR   = vars["RANGE_DIR"]
        isAlreadyExecuted = True
