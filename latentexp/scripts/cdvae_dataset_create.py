import os
import json
from glob import glob

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.dimensionality import get_dimensionality_cheon, get_dimensionality_gorai

from latentexp.utils import MP_DIR, JSON_DIR, N_CORES
from multiprocessing import Pool

# Define a function to process each file
def process_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    # Convert structure to pymatgen structure
    structure = Structure.from_dict(data['structure'])

    if get_dimensionality_gorai(structure) == 3:
        # Use CifWriter to get the CIF content as a string
        cif_writer = CifWriter(structure)

        return data['material_id'], data['formula_pretty'], data['elements'], cif_writer.cif_file
    else:
        return None, None, None, None

def process_data(save_dir):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    files=glob(os.path.join(JSON_DIR,'*.json'))

    df_dict={'material_id':[],
            'pretty_formula':[],
            'elements':[],
            'cif':[]}

    # Create a multiprocessing Pool with the desired number of processes
    with Pool(N_CORES) as pool:
        # Map the process_file function to each file in parallel
        results=pool.map(process_file, files)

    # Loop through results and remove any None entries. Store them in the df_dict
    for result in results:
        if result[0] is not None:
            df_dict['material_id'].append(result[0])
            df_dict['pretty_formula'].append(result[1])
            df_dict['elements'].append(result[2])
            df_dict['cif'].append(result[3])

    # Save the data to a pandas dataframe
    df=pd.DataFrame(df_dict)

    # Split df into training,test, and validation sets
    train_df=df.sample(frac=0.8,random_state=0)
    test_df=df.drop(train_df.index)
    val_df=train_df.sample(frac=0.1,random_state=0)
    train_df=train_df.drop(val_df.index)

    # Save the training, test, and validation sets to csv files
    train_df.to_csv(os.path.join(save_dir,'train.csv'),index=False)
    test_df.to_csv(os.path.join(save_dir,'test.csv'),index=False)
    val_df.to_csv(os.path.join(save_dir,'val.csv'),index=False)

    # Save the main dataframe to a csv file
    df.to_csv(os.path.join(save_dir,'mp_database.csv'),index=False)


if __name__ == "__main__":

    save_dir = os.path.join(MP_DIR, '3d')
    process_data(save_dir)