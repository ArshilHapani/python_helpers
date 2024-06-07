# Machine Learning Python Helpers

This repository contains a collection of helper functions for machine learning tasks. These functions are designed to assist in automating common tasks and streamlining the machine learning workflow.

## Installation

To use these helper functions in your own projects, you can clone this repository using the following command:

## Helper function for downloading function directly into colab

```py
# download helper function from remote repo if it's not already downloaded
import requests
from pathlib import Path
import re



def download_file(url:str, name:str)->None:
    """
    Downloads a file from a raw GitHub URL if the file does not already exist.

    The function validates the URL and filename before proceeding with the download.
    If the URL is not a raw GitHub URL or if the filename does not have a .py extension,
    it prints an error message and returns without downloading the file.

    Parameters:
    url (str): The URL of the file to download. Must be a raw GitHub URL.
    name (str): The name to give to the downloaded file. Must have a .py extension.

    Returns:
    N
    """
    if not re.match(r"https://raw.githubusercontent.com/.+/.+/.+\.py", url):
        print("Invalid URL. Please provide a raw GitHub URL.")
        return

    if not name.endswith(".py"):
        print("Invalid filename. Please provide a filename with a .py extension.")
        return

    if Path(name).is_file():
        print(f"{name} already exists, skipping download.")
        return
    else:
        print(f"Downloading {name} from {url}...")
        request = requests.get(url)

        with open(name, "w") as file:
            file.write(request.text)
            print("File downloaded...")

```

Copy the above function and paste it in your colab notebook and use it to download the helper function from this repo or any other repo.

Checkout `downloadFile.py` for raw function code.

**Note**: My indentation may be different then yours so please make sure to correct it before using it.

# function list

- to_tensor
- to_cpu
- training_testing_loop_classification_model
- plot_decision_boundary
- print_train_time
