# -*- coding: utf-8 -*-
"""
Handles the data saving to various formats.
"""
import pandas as pd

class DataSaver:
    """
    Saves simulation data to files.
    """
    def __init__(self, filepath):
        """
        Initializes the DataSaver.

        Args:
            filepath (str): The path to the output file.
        """
        self.filepath = filepath

    def save_to_csv(self, data):
        """
        Saves a list of dictionaries to a CSV file.

        Args:
            data (list): A list of data records (dictionaries).
        """
        if not data:
            return
        
        df = pd.DataFrame(data)
        df.to_csv(self.filepath, index=False)
        print(f"Data saved to {self.filepath}")

    def save_report(self, report_content, report_path):
        """
        Saves a text report to a file.

        Args:
            report_content (str): The content of the report.
            report_path (str): The path to save the report.
        """
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"Report saved to {report_path}")
