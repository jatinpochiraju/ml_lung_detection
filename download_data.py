"""
Download chest X-ray datasets from Kaggle for:
- COVID-19
- Pneumonia  
- Tuberculosis (TB)
- Healthy/Normal

Using the combined dataset: jtiptj/chest-xray-pneumoniacovid19tuberculosis
"""
import os
import subprocess
import zipfile
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

def download_dataset():
    """Download the chest X-ray dataset with COVID, Pneumonia, TB, and Normal classes."""
    
    # Dataset: chest-xray-pneumoniacovid19tuberculosis by jtiptj
    # Contains 4 classes: COVID19, PNEUMONIA, TUBERCULOSIS, NORMAL
    dataset = "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
    
    print(f"Downloading dataset: {dataset}")
    print("This may take a few minutes...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    result = subprocess.run(
        ["python", "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", DATA_DIR, "--unzip"],
        capture_output=True, text=True, timeout=600
    )
    
    print("STDOUT:", result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("\nTrying alternative dataset...")
        # Fallback: tawsifurrahman/covid19-radiography-database
        dataset2 = "tawsifurrahman/covid19-radiography-database"
        result2 = subprocess.run(
            ["python", "-m", "kaggle", "datasets", "download", "-d", dataset2, "-p", DATA_DIR, "--unzip"],
            capture_output=True, text=True, timeout=600
        )
        print("STDOUT:", result2.stdout)
        if result2.returncode != 0:
            print("STDERR:", result2.stderr)
            return False
    
    return True

def verify_data():
    """List what was downloaded."""
    print("\n--- Downloaded Data Structure ---")
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 3:
            subindent = ' ' * 2 * (level + 1)
            file_count = len(files)
            if file_count > 5:
                for f in files[:3]:
                    print(f'{subindent}{f}')
                print(f'{subindent}... and {file_count - 3} more files')
            else:
                for f in files:
                    print(f'{subindent}{f}')

if __name__ == "__main__":
    success = download_dataset()
    if success:
        verify_data()
        print("\nDataset download complete!")
    else:
        print("\nFailed to download dataset. Please check your Kaggle credentials.")
