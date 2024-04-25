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

def download_file(url, name):
     """
        Downloads a file from a raw GitHub URL if the file does not already exist.

        The function validates the URL and filename before proceeding with the download. 
        If the URL is not a raw GitHub URL or if the filename does not have a .py extension, 
        it prints an error message and returns without downloading the file.

        Parameters:
        url (str): The URL of the file to download. Must be a raw GitHub URL.
        name (str): The name to give to the downloaded file. Must have a .py extension.

        Returns:
        None
    """
    # Check if the URL is from raw GitHub
    if not re.match(r'https://raw\.githubusercontent\.com/.+', url):
        print("Invalid URL. Please provide a raw GitHub URL.")
        return

    # Check if the filename has a .py extension
    if not name.endswith('.py'):
        print("Invalid filename. Please provide a filename with a .py extension.")
        return

    if Path(name).is_file():
        print(f"{name} already exist, skipping download")
    else:
        print(f"Downloading {name}")
        request = requests.get(url)

        with open(name,"wb") as f:
            f.write(request.content)
```

Copy the above function and paste it in your colab notebook and use it to download the helper function from this repo or any other repo.
