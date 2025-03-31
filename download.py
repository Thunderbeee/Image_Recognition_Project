import os
import requests
import zipfile
from tqdm import tqdm
import concurrent.futures

BASE_URL = "https://tdface.ece.tufts.edu/downloads/TD_RGB_E/"
DATASET_FILES = [
    "TD_RGB_E_Set1.zip",
    "TD_RGB_E_Set2.zip",
    "TD_RGB_E_Set3.zip",
    "TD_RGB_E_Set4.zip"
]

DOWNLOAD_DIR = "data/downloaded"
EXTRACT_DIR = "data/extracted"
MAX_WORKERS = 16

def download_file(filename):
    url = BASE_URL + filename
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"File {filepath} already exists, skipping download.")
        return filepath
    
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return None
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    print(f"Downloading {url} to {filepath}")
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    
    with open(filepath, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    return filepath

def extract_zip(zip_filepath):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    zip_filename = os.path.basename(zip_filepath)
    print(f"Extracting {zip_filename} to {EXTRACT_DIR}")
    
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            
            zip_ref.extractall(EXTRACT_DIR)
            print(f"Extraction complete: {zip_filename}, {total_files} files")
        return True
    except Exception as e:
        print(f"Error extracting {zip_filepath}: {str(e)}")
        return False

def main():

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    print(f"Starting parallel download of {len(DATASET_FILES)} datasets...")
    
    downloaded_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(download_file, filename): filename for filename in DATASET_FILES}
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                path = future.result()
                if path:
                    downloaded_paths.append(path)
                    print(f"Download complete: {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
    
    print(f"Downloaded {len(downloaded_paths)} out of {len(DATASET_FILES)} files")
    
    print("Starting parallel extraction of all files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_zip = {executor.submit(extract_zip, zip_path): zip_path for zip_path in downloaded_paths}
        for future in concurrent.futures.as_completed(future_to_zip):
            zip_path = future_to_zip[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error extracting {os.path.basename(zip_path)}: {str(e)}")
    
    print("All datasets downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
