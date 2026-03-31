"""
Download chest X-ray datasets from Kaggle using the API directly.
Classes: COVID-19, Pneumonia, Tuberculosis (TB), Normal/Healthy
"""
import os
import json
import zipfile

os.environ['KAGGLE_USERNAME'] = 'jatinpochiraju'
os.environ['KAGGLE_KEY'] = 'KGAT_28f10b6238c35363977e66f549618f0a'

from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

def download_dataset():
    api = KaggleApi()
    api.authenticate()
    
    dataset = "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
    print(f"Downloading dataset: {dataset}")
    print("This may take several minutes depending on your connection...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)
        print("Download and extraction complete!")
        return True
    except Exception as e:
        print(f"Error with primary dataset: {e}")
        print("Trying alternative dataset...")
        try:
            dataset2 = "tawsifurrahman/covid19-radiography-database"
            api.dataset_download_files(dataset2, path=DATA_DIR, unzip=True)
            print("Alternative dataset downloaded!")
            return True
        except Exception as e2:
            print(f"Error with alternative dataset: {e2}")
            return False

def verify_data():
    print("\n--- Downloaded Data Structure ---")
    total_images = 0
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f'{indent}{os.path.basename(root)}/ ({len(img_files)} images)')
        total_images += len(img_files)
    print(f"\nTotal images found: {total_images}")

if __name__ == "__main__":
    success = download_dataset()
    if success:
        verify_data()
        print("\nDataset ready for training!")
    else:
        print("\nFailed to download dataset.")
