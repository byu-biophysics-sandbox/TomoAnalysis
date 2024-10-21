import os
import json
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert non-serializable objects to serializable ones
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        # Add more conversions here if needed
        return super().default(obj)

class DictSaver:
    def __init__(self, output_file="./myfile.json"):
        self.file_path = output_file
        self.data = {}
        self._load_existing_data()

    def _load_existing_data(self):
        """
        Load existing data from the JSON file if it exists.
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as json_file:
                    self.data = json.load(json_file)
                print(f"Data loaded from {self.file_path}.")
            except json.JSONDecodeError:
                print(f"The file {self.file_path} is empty or contains invalid JSON. Starting with an empty dictionary.")
                self.data = {}
        else:
            print(f"No existing file found. A new file will be created at {self.file_path}.")
            self.data = {}

    def write(self, results: dict, **kwargs):
        try:
            # Update the existing data with the new data
            self.data.update(results)

            # Write the updated data back to the file using the custom encoder
            with open(self.file_path, 'w') as json_file:
                json.dump(self.data, json_file, indent=4, cls=CustomJSONEncoder)

            print(f"Data appended to {self.file_path} successfully.")
        except Exception as e:
            print(f"An error occurred while writing to {self.file_path}: {e}")

    def finish_and_save(self):
        """
        Explicitly save the accumulated data to the JSON file.
        """
        try:
            with open(self.file_path, 'w') as json_file:
                json.dump(self.data, json_file, indent=4, cls=CustomJSONEncoder)
            print(f"Data saved to {self.file_path} successfully.")
        except Exception as e:
            print(f"An error occurred while saving to {self.file_path}: {e}")