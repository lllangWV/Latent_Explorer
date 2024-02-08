from pymatgen.analysis.dimensionality import get_dimensionality_cheon, get_dimensionality_gorai
from pymatgen.core import Structure

structure_2D = Structure.from_file('sandbox/2d.poscar')

structure_3D = Structure.from_file('sandbox/3d.poscar')

print(structure_2D)




print(get_dimensionality_cheon(structure_2D))
print(get_dimensionality_cheon(structure_3D))


print(get_dimensionality_gorai(structure_2D))
print(type(get_dimensionality_gorai(structure_3D)))

# Another method from ase

# from ase.build import graphene, mx2, bulk
# from ase.geometry.dimensionality import analyze_dimensionality

# def get_dim(atoms, method='RDA'):
#     return analyze_dimensionality(atoms, method=method)[0].dimtype

# # Graphene monolayer
# graphene_atoms = graphene()
# dim1 = get_dim(graphene_atoms)
# print(dim1) # 2D

# # 3-layer MoS2 
# MoS2_atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
# dim2 = get_dim(MoS2_atoms)
# print(dim2) # 2D

# # Cu bulk
# Cu_atoms = bulk('Cu', 'fcc', a=4.0)
# dim3 = get_dim(Cu_atoms)
# print(dim3) # 3D