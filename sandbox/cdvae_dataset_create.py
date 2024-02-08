import os
import json
from glob import glob

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from latentexp.utils import MP_DIR,JSON_DIR


files=glob(os.path.join(JSON_DIR,'*.json'))

print(files[:3])


df_dict={'material_id':[],
         'pretty_formula':[],
         'elements':[],
         'cif':[]}

# Loop over the files and load the data
for file in files:

    with open(file, 'r') as f:
        data = json.load(f)

        # Convert structure to pymatgen structure
        structure = Structure.from_dict(data['structure'])

        # Use CifWriter to get the CIF content as a string
        cif_writer = CifWriter(structure)

        # Append the data to the dictionary
        df_dict['material_id'].append(data['material_id'])
        df_dict['pretty_formula'].append(data['formula_pretty'])
        df_dict['elements'].append(data['elements'])
        df_dict['cif'].append(cif_writer.cif_file)

# Save the data to a pandas dataframe
df=pd.DataFrame(df_dict)

# Split df into training,test, and validation sets
train_df=df.sample(frac=0.8,random_state=0)
test_df=df.drop(train_df.index)
val_df=train_df.sample(frac=0.2,random_state=0)
train_df=train_df.drop(val_df.index)


# Save the dataframes to csv files
train_df.to_csv(os.path.join(MP_DIR,'train.csv'),index=False)
test_df.to_csv(os.path.join(MP_DIR,'test.csv'),index=False)
val_df.to_csv(os.path.join(MP_DIR,'val.csv'),index=False)


# Save the main dataframe to a csv file
df.to_csv(os.path.join(MP_DIR,'mp_database.csv'),index=False)

