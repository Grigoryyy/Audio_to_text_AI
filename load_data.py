import os
import tarfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

def extract_tar_gz(filepath, extract_to='.'):
    with tarfile.open(filepath, 'r:gz') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc='Extracting') as pbar:
            for member in members:
                tar.extract(member, path=extract_to)
                pbar.update(1)

def main():
    datasets = [
        {
            'name': 'radio_pspeech_sample_manifest',
            'url': 'https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/radio_pspeech_sample_manifest.tar.gz'
        },
        {
            'name': 'radio_2', 
            'url': 'https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/radio_2.tar.gz'
        }
    ]
    
    for dataset in datasets:
        filename = f"{dataset['name']}.tar.gz"
        
        if not os.path.exists(filename):
            download_file(dataset['url'], filename)
        
        extract_folder = dataset['name']
        
        if not os.path.exists(extract_folder):
            extract_tar_gz(filename, extract_folder)

if __name__ == "__main__":
    main()
