from writers import Writer
import pandas as pd
import os

class CSV(Writer):
    def __init__(self, output_file="./myfile.csv"):
        self.file_path = output_file
        self.df = pd.DataFrame()  # Initialize an empty DataFrame to accumulate results
        self._load_existing_data()

    def _load_existing_data(self):
        """
        Load existing data from the CSV file if it exists.
        """
        if os.path.exists(self.file_path):
            try:
                self.df = pd.read_csv(self.file_path)
            except pd.errors.EmptyDataError:
                print(f"The file {self.file_path} is empty. Starting with a new DataFrame.")
        else:
            print(f"No existing file found. A new file will be created at {self.file_path}.")

    def write(self, results: dict, **kwargs):
        """
        Write results to the DataFrame and prepare to save to CSV.
        
        Args:
        - results (dict): Dictionary where keys are row identifiers and values are dicts representing rows of data.
        """
        rows = []
        
        # Determine all possible columns from the entire dataset
        all_columns = set()
        for attrs in results.values():
            all_columns.update(attrs.keys())
        all_columns = sorted(all_columns)  # Sort columns to maintain order
        
        for path, attrs in results.items():
            row = {'file_path': path}
            for column in all_columns:
                row[column] = attrs.get(column, None)  # Use None if column is missing in the attributes
            rows.append(row)
        
        # Convert rows to a DataFrame
        new_df = pd.DataFrame(rows)
        
        # Append new data to the existing DataFrame
        self.df = pd.concat([self.df, new_df], ignore_index=True)
         
    def finish_and_save(self):
        """
        Save the accumulated DataFrame to the CSV file.
        """
        self.df.to_csv(self.file_path, index=False)
        print(f"Data saved to {self.file_path}")

if __name__ == "__main__":
    writer = CSV()

    writer.write()