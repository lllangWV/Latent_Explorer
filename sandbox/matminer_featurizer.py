import os
import ast
import pandas as pd
from pymatgen.core import Structure
from latentexp.utils import DATA_DIR

csv_file=os.path.join(DATA_DIR,'processed','qm9','targets_gap.csv')

df = pd.read_csv(csv_file, nrows=2)
df['Structure'] = df['Structure'].apply(lambda x: Structure.from_dict(ast.literal_eval(x)))
print(df)



