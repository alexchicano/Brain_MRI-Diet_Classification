import pandas as pd
import numpy as np

class DatasetMetadata():
    def __init__(self, dataset, modality, interval = [], csv_path = '../data/metadata/metadata.csv'):
        '''Initialize the dataset object and its attributes'''
        
        # General info about the dataset:
        self.dataset = dataset
        self.modality = modality
        
        # Select info from the csv file:
        df = pd.read_csv(csv_path, sep=';')
        self.df = df
        
        self.df = self.df.sort_values(by=['ID']) # Sort by ID
        self.df = self.df[self.df['dataset'] == dataset] #Filter by dataset name
        self.df = self.df[self.df[modality+'_exists'] == True] #Remove those cases without the chosen modality
        self.df = self.df[np.invert(np.isnan(self.df['score']))] #Remove those cases without a score
        
        # If interval applied, filter by range(s) of values:
        if len(interval) != 0:
            if len(np.shape(interval)) == 1: # If only one interval is given:
                self.df = self.df[self.df['score'] >= interval[0]]
                self.df = self.df[self.df['score'] <= interval[1]]
            else: # If multiple intervals are given:
                in_range = 0
                for i in range(len(interval)):
                    # keep only the cases that are in the interval range:
                    in_range += (self.df['score'] >= interval[i][0]) & (self.df['score'] <= interval[i][1])
                    
                self.df = self.df[in_range >= 1]
                    
        # Get the IDs and Score list and the number of cases:
        self.IDs = list(self.df['ID'])
        
        self.len = len(self.IDs)
        self.scores = list(self.df['score'])
    
        # Returns a dictionary with the IDs as key and the rows of df as values:
        self.id_dict = self.df.set_index('ID').to_dict('index')
