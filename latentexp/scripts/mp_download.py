import os
import shutil
import json

from mp_api.client import MPRester

from latentexp.utils import DATA_DIR, MP_API_KEY


FIELDS_TO_INCLUDE=['material_id','nsites','elements','nelements','composition',
                   'composition_reduced','formula_pretty','volume',
                   'density','density_atomic','symmetry','structure',
                   'energy_per_atom','formation_energy_per_atom','energy_above_hull','is_stable',
                   'band_gap','cbm','vbm','is_stable','efermi','is_gap_direct','is_metal',
                   'is_magnetic','ordering','total_magnetization','total_magnetization_normalized_vol',
                   'num_magnetic_sites','num_unique_magnetic_sites',
                   'k_voigt','k_reuss','k_vrh','g_voigt','g_reuss','g_vrh',
                   'universal_anisotropy','homogeneous_poisson','e_total','e_ionic','e_electronic']


def screen_3d_materials(criteria):
    # Get possible materials based on criteria
    with MPRester(MP_API_KEY) as mpr:
        summary_docs = mpr.materials.summary._search( **criteria, fields=['material_id','structure'])

    material_ids=[]
    structures=[]
    for doc in summary_docs:
        material_ids.append(str(doc.material_id))
        structures.append(doc.structure)

    # Screen 3d dimensional material through robocrys api endpoint. Defines 3d and 2d materials.
    filtered_material_ids=[]
    filtered_structures=[]
    for i,material_id in enumerate(material_ids):
        try:
            robocrys_docs = mpr.robocrys._search( material_ids = [material_id], fields = ['condensed_structure'])[0]
            if robocrys_docs.condensed_structure.dimensionality == 3:
                filtered_material_ids.append(material_id)
                filtered_structures.append(structures[i])
        except Exception as e:
            continue

    n_3d_material = len(filtered_material_ids)
    print("Found {0} possible 3d materials".format(n_3d_material))
    return filtered_material_ids

def get_mp_materials(mp_dir,criteria, screen_3d=False):

    # Create directories
    MP_DIR=mp_dir
    JSON_DIR=os.path.join(MP_DIR,'json_database')
    os.makedirs(MP_DIR,exist_ok=True)
    os.makedirs(JSON_DIR,exist_ok=True)

    # Save criteria to json file in MP_DIR
    with open(os.path.join(MP_DIR,'criteria.json'), 'w') as f:
        json.dump(criteria, f, indent=4)

    # Screen 3d dimensional material
    if screen_3d:
        filtered_material_ids=screen_3d_materials(criteria)
        with MPRester(MP_API_KEY) as mpr:
            summary_docs = mpr.summary._search( material_ids=filtered_material_ids, fields=FIELDS_TO_INCLUDE,chunk_size=500)
        
    else:
        with MPRester(MP_API_KEY) as mpr:
            summary_docs = mpr.materials.summary._search(**criteria,
                                            fields=FIELDS_TO_INCLUDE)
        
    # Process selected materials
    for doc in summary_docs:
        summary_doc_dict = doc.dict() 
        mp_id=summary_doc_dict['material_id']

        json_file=os.path.join(JSON_DIR,f'{mp_id}.json')

        json_database_entry={}
        for field_name in FIELDS_TO_INCLUDE:
            try:
                json_database_entry.update({field_name:summary_doc_dict[field_name]})
            except:
                json_database_entry.update({field_name:None})

        if not os.path.exists(os.path.dirname(json_file)):
            os.makedirs(os.path.dirname(json_file))
            
        with open(json_file, 'w') as f:
            json.dump(json_database_entry, f, indent=4)
    print('------------------------------------------------')




if __name__ == "__main__":

    # # For all materials
    # mp_dir=os.path.join(DATA_DIR,'raw',f'mp_database')
    # # Find other endpoints here https://api.materialsproject.org/docs#/Molecules%20Summary/search_molecules_summary__get
    # criteria={'nelements_min':2,
    #         'nelements_max':5,
    #         'nsites_max':20,
    #         'energy_above_hull_min':0,
    #         'energy_above_hull_max':0.05}
    # get_mp_materials(mp_dir,criteria)


    # For all materials
    mp_dir=os.path.join(DATA_DIR,'raw',f'mp_database_3d')
    # # Find other endpoints here https://api.materialsproject.org/docs#/Molecules%20Summary/search_molecules_summary__get
    # criteria={'nelements_min':2,
    #         'nelements_max':5,
    #         'nsites_max':20,
    #         'energy_above_hull_min':0,
    #         'energy_above_hull_max':0.05}
    # get_mp_materials(mp_dir,criteria,screen_3d=True)