from collections import Counter
import argparse
import os
import json
import time

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty



from torch.optim import Adam
from types import SimpleNamespace
from torch_geometric.data import Batch
from scipy.stats import norm

from eval_utils import load_model


from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)

def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list

def generation(loader, model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []


    batch=next(iter(loader))



    if torch.cuda.is_available():
        batch.cuda()

    # only sample one z, multiple evals for stoichaticity in langevin dynamics
    _, _, z = model.encode(batch)

    # Only use first sample in the batch
    first_sample=z[:1,:]


    index_to_replace = 8
    mean, std = 0, 1  # Standard normal distribution parameters
    num_samples = 10
    samples = np.linspace(mean - 3*std, mean + 3*std, num_samples)
    print(first_sample)
    print(first_sample.shape)
    # Clone and modify the tensor
    cloned_tensors = first_sample.repeat(num_samples+1,1)
    print(cloned_tensors)
    print(cloned_tensors.shape)
    for i, sample in enumerate(samples):
        cloned_tensors[i+1, index_to_replace] = sample

    print(cloned_tensors)

    batch_all_frac_coords = []
    batch_all_atom_types = []
    batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
    batch_lengths, batch_angles = [], []

    samples = model.langevin_dynamics(cloned_tensors, ld_kwargs)

    # collect sampled crystals in this batch.
    batch_frac_coords.append(samples['frac_coords'].detach().cpu())
    batch_num_atoms.append(samples['num_atoms'].detach().cpu())
    batch_atom_types.append(samples['atom_types'].detach().cpu())
    batch_lengths.append(samples['lengths'].detach().cpu())
    batch_angles.append(samples['angles'].detach().cpu())
    if ld_kwargs.save_traj:
        batch_all_frac_coords.append(
            samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
        batch_all_atom_types.append(
            samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

    # collect sampled crystals for this z.
    frac_coords.append(torch.stack(batch_frac_coords, dim=0))
    num_atoms.append(torch.stack(batch_num_atoms, dim=0))
    atom_types.append(torch.stack(batch_atom_types, dim=0))
    lengths.append(torch.stack(batch_lengths, dim=0))
    angles.append(torch.stack(batch_angles, dim=0))
    if ld_kwargs.save_traj:
        all_frac_coords_stack.append(
            torch.stack(batch_all_frac_coords, dim=0))
        all_atom_types_stack.append(
            torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)




def main(args):


    model_path = Path(args.root_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate model on the generation task.')
    start_time = time.time()

    (frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack) = generation(test_loader,
        model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
        args.batch_size, args.down_sample_traj_step)


    torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / 'frozen.pt')


    crys_array_list, _ = get_crystal_array_list( str(model_path / 'frozen.pt'), batch_idx=0)
    gen_crys = p_map(lambda x: Crystal(x), crys_array_list)


    generated_dir=os.path.join(args.root_path, 'generated_frozen_8')
    os.makedirs(generated_dir,exist_ok=True)

    for i,crys in enumerate(gen_crys):
        struct=crys.structure
  
        filename=os.path.join(generated_dir, f'POSCAR_{i}')
        struct.to(fmt="poscar", filename=filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()
    main(args)
