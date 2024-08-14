import os

def get_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key