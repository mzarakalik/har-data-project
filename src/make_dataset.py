import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import urllib.request
import zipfile

class DataPreparation:
    def __init__(self, raw_data_path='data/raw', processed_data_path='data/processed'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
    def download_dataset(self):
        """Download UCI HAR Dataset"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = os.path.join(self.raw_data_path, "dataset.zip")
        
        # Create directory if it doesn't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        # Download the dataset
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_data_path)
            
        # Remove zip file
        os.remove(zip_path)
        
    def load_data(self):
        """Load and merge the dataset"""
        base_path = os.path.join(self.raw_data_path, "UCI HAR Dataset")
        
        # Load training data
        X_train = pd.read_csv(os.path.join(base_path, "train/X_train.txt"), sep=r"\s+", header=None)
        y_train = pd.read_csv(os.path.join(base_path, "train/y_train.txt"), header=None)
        
        # Load test data
        X_test = pd.read_csv(os.path.join(base_path, "test/X_test.txt"), sep=r"\s+", header=None)
        y_test = pd.read_csv(os.path.join(base_path, "test/y_test.txt"), header=None)
        
        # Load feature names
        features = pd.read_csv(os.path.join(base_path, "features.txt"), sep=r"\s+", header=None)[1]
        
        # Add unique suffix to duplicate column names
        feature_names = []
        seen = set()
        for name in features:
            if name in seen:
                count = 1
                new_name = f"{name}_{count}"
                while new_name in seen:
                    count += 1
                    new_name = f"{name}_{count}"
                feature_names.append(new_name)
            else:
                feature_names.append(name)
            seen.add(feature_names[-1])
            
        # Assign unique column names
        X_train.columns = feature_names
        X_test.columns = feature_names
        y_train.columns = ['activity']
        y_test.columns = ['activity']
        
        return X_train, X_test, y_train, y_test
        
    def preprocess_data(self):
        """Preprocess the dataset"""
        # Download if data doesn't exist
        if not os.path.exists(os.path.join(self.raw_data_path, "UCI HAR Dataset")):
            self.download_dataset()
            
        # Load data
        print("Loading data...")
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Create processed data directory
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Save processed datasets
        print("Saving processed data...")
        X_train.to_parquet(os.path.join(self.processed_data_path, "X_train.parquet"))
        X_test.to_parquet(os.path.join(self.processed_data_path, "X_test.parquet"))
        y_train.to_parquet(os.path.join(self.processed_data_path, "y_train.parquet"))
        y_test.to_parquet(os.path.join(self.processed_data_path, "y_test.parquet"))
        
        print("Data preprocessing completed!")

if __name__ == "__main__":
    data_prep = DataPreparation()
    data_prep.preprocess_data()