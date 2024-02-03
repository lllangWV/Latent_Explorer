import os
from glob import glob
import csv

import pandas as pd
import numpy as np 
from pymatgen.core import Structure

from latentexp.utils import LOGGER, DATA_DIR
import numpy as np

def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split('*^')
        return float(base) * 10**float(power)


def parse_xyz(filename):
    """
    Parses QM9 specific xyz files. See https://www.nature.com/articles/sdata201422/tables/2 for reference
    :param filename: str path to file
    :return:
    """
    num_atoms = 0
    scalar_properties = []
    atomic_symbols = []
    xyz = []
    charges = []
    harmonic_vibrational_frequencies = []
    smiles = ''
    inchi = ''
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                num_atoms = int(line)
            elif line_num == 1:
                scalar_properties = [float(i) for i in line.split()[2:]]
            elif 2 <= line_num <= 1 + num_atoms:
                atom_symbol, x, y, z, charge = line.split()
                atomic_symbols.append(atom_symbol)
                xyz.append([parse_float(x), parse_float(y), parse_float(z)])
                charges.append(parse_float(charge))
            elif line_num == num_atoms + 2:
                harmonic_vibrational_frequencies = [float(i) for i in line.split()]
            elif line_num == num_atoms + 3:
                smiles = line.split()[0]
            elif line_num == num_atoms + 4:
                inchi = line.split()[0]

    result = {
        'num_atoms': num_atoms,
        'atomic_symbols': atomic_symbols,
        'pos': np.array(xyz),
        'charges': np.array(charges),
        'harmonic_oscillator_frequencies': harmonic_vibrational_frequencies,
        'smiles': smiles,
        'inchi': inchi
    }
    scalar_property_labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u', 'h', 'g', 'cv']
    scalar_properties = dict(zip(scalar_property_labels, scalar_properties))

    result.update({'scalar_properties':scalar_properties})
    return result


def process_xyz_files(dir, save_dir):
    structures = []
    targets = []

    files=glob(os.path.join(dir, "*.xyz"))
    # Iterate over all files in the directory

    # Get number of scalar proerties present in the files
    result=parse_xyz(files[0])
    n_scalar_values=len(result['scalar_properties'])


    structures=[None]*len(files)
    targets_list=np.zeros(shape=(len(files),n_scalar_values))
    for i,filename in enumerate(files):
        if i%5000==0:
            print(i)
            
        result=parse_xyz(filename)

        # Creat pymatgen Structure object
        lattice = 16*np.diag(3*[1])
        frac_coords=np.dot(result['pos'], np.linalg.inv(lattice))
        species=result['atomic_symbols']
        struct=Structure(lattice=lattice, species=species,coords=frac_coords,)
        structures[i] = struct.as_dict()

        # Put scalar properties in array
        target_names = list(result['scalar_properties'].keys())
        targets = [ value for key, value in result['scalar_properties'].items()]
        
        targets_list[i,:]=np.array(targets)

    for j in range(n_scalar_values):
        targets=targets_list[:,j]
        tmp_dict={'Structure':structures,'Target':targets}
        df=pd.DataFrame(tmp_dict)
        df.to_csv(os.path.join(save_dir,f'targets_{target_names[j]}.csv'),index=False)
    return df

if __name__ == "__main__":
    # Example usage

    dir=os.path.join(DATA_DIR,'raw','QM9')
    save_dir=os.path.join(DATA_DIR,'processed','qm9')
    process_xyz_files(dir=dir, save_dir=save_dir)
