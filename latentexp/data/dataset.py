import os
import ast

from pymatgen.core import Structure
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import XRDPowderPattern
from matminer.featurizers.composition import ElementFraction


def get_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


class MyDataset(Dataset):
    def __init__(self, data):

        self.csv_file=csv_file

        self.data = self.get_data(self.csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # Process the sample if needed
        # ...

        return sample
    
    def get_data(self,csv_file):
        df = pd.read_csv(csv_file)

        # Convert the 'Structure' column to pymatgen.core.Structure objects
        df['Structure'] = df['Structure'].apply(lambda x: Structure.from_dict(ast.literal_eval(x)))

        # Convert the structures and compositions to pandas dataframe. This is required to use Matminer featurizers
        # structure_data = pd.DataFrame({'structure': structures},)
        
        # structure_featurizer = MultipleFeaturizer([XRDPowderPattern()])
        # composition_featurizer = MultipleFeaturizer([ElementFraction()])

        # structure_features = structure_featurizer.featurize_dataframe(structure_data,"structure")
        # composition_features = composition_featurizer.featurize_dataframe(composition_data, "composition")

        # structure_features=structure_features.drop(columns=['structure'])
        # composition_features=composition_features.drop(columns=['composition'])
        y=df['Target']
        x=df['Structure']
        return x, y 
    

    def save(self):
        torch.save(self, 'dataset.pth')

    def load(self):
        torch.load('dataset.pth')
    
# class CsvLoader():
#     def __init__(self, csv_file, batch_size=1, shuffle=False, num_workers=0):
#         # super(MyDataLoader, self).__init__(
#         #     dataset=dataset,
#         #     batch_size=batch_size,
#         #     shuffle=shuffle,
#         #     num_workers=num_workers
#         # )
#         self.csv_file=csv_file

#         self.data=self.get_data(self.csv_file)



#     def get_data(self, csv_file):
#         df = pd.read_csv(csv_file)

#         # Convert the 'Structure' column to pymatgen.core.Structure objects
#         df['Structure'] = df['Structure'].apply(lambda x: Structure.from_dict(ast.literal_eval(x)))

#         print(df.head())
        
#         return df
    

if __name__=="__main__":
    # Create a dataset
    # data = list(range(100))
    # dataset = MyDataset(data)

    # Create a dataloader
    # dataloader = MyDataLoader(dataset, batch_size=32, shuffle=True)

    # # # Iterate over the dataloader
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch {batch_idx}: {batch}")
    #     # Do something with the batch
        # ...
    from latentexp.utils import DATA_DIR

    csv_file=os.path.join(DATA_DIR,'processed','qm9','targets_gap.csv')
    

    dataset=MyDataset(csv_file)
    # csv_loader=CsvLoader(csv_file)


