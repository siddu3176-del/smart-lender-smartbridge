import urllib.request
import os

def download_dataset():
    url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
    output_dir = "data"
    output_path = os.path.join(output_dir, "loan_prediction.csv")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Dataset downloaded successfully to {output_path}")

if __name__ == "__main__":
    download_dataset()
