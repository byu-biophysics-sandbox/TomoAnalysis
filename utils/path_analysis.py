import os
import difflib                     # Import difflib for sequence matching

class PathAnalysis:
    def __init__(self, path):
        self.path = path
    
    # Function to check if file path indicates a SIRT tomogram
    def is_sirt(self):
        """
        Check if the file path indicates a SIRT tomogram.
        
        Args:
        - file: File path of the tomogram file.
        
        Returns:
        - bool: True if "sirt" is in the file name, False otherwise.
        """
        if 'sirt' in os.path.basename(self.path).lower():
            return True
        else:
            return False

    # Function to find the name of bacteria from the file path
    def find_bacteria_name(self):
        """
        Find the name of bacteria from the file path.
        
        Args:
        - path: File path containing information about the bacteria.
        
        Returns:
        - list: List of possible names of bacteria found in the file path.
        """
        possibilities = []
        with open("/home/matiasgp/Desktop/Toma-Analysis/analyzers/cleaned_titles_input.txt", 'r') as infile:
            possibilities = [line.strip() for line in infile]
        words = []
        for w in self.path.split("/"):
            # Use difflib to find close matches of path components in possibilities
            if len(difflib.get_close_matches(w, possibilities, n=1, cutoff=0.51)) != 0:
                words.append(w)
        if len(words) == 0:
            # If no matches found, attempt to retrieve name from specific path components
            words = self.path.split("/")[9]
            if "tomodb1_d4" in self.path.split("/")[7].lower:
                words = self.path.split("/")[8]
        return words