import os
import pickle

import numpy as np
import torch
import trimesh
from Bio.PDB import PDBParser, Selection
from tqdm import tqdm

from src.datamodules.datasets.chem.computeCharges import computeChargeHelper, computeSatisfied_CO_HN
from src.datamodules.datasets.chem.computeHydro import computeHydrophobicity
from src.datamodules.datasets.chem.constants import BBHeavyAtom

Tensor, tensor = torch.LongTensor, torch.FloatTensor


def computer_property(pdb_path, pts, surf_mode):
    # compute chemical/biological properties
    parser = PDBParser()
    structure = parser.get_structure("tmp", pdb_path)
    residues, atom_xyz, atom_types, res_coords, res_types = [], [], [], [], []
    for i, res in enumerate(structure.get_residues()):
        residues.append(res)
        res_types.append(res.get_resname())
        for atom in res.get_atoms():
            if atom.name in BBHeavyAtom._member_names_:
                atom_xyz.append(atom.get_coord())
                atom_types.append(atom.element)
            if atom.name == 'CA':
                res_coords.append(atom.get_coord())
    atom_xyz, res_coords, res_types, atom_types = tensor(np.stack(atom_xyz)), tensor(np.stack(res_coords)), np.array(res_types), np.array(atom_types)

    if surf_mode == 'pymol':
        delta_pos = pts.mean(0) - atom_xyz.mean(0)
        if delta_pos.abs().max() > 3:
            print(delta_pos)

    # compute hydrophobicity
    knn_res_idx = torch.cdist(pts, res_coords).topk(1, dim=1, largest=False)[1].squeeze()
    knn_res_types = res_types[knn_res_idx].tolist()
    hp: Tensor = computeHydrophobicity(knn_res_types)

    # compute charge
    hbond: Tensor = torch.zeros(len(pts))
    atoms = Selection.unfold_entities(structure, "A")
    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)
    knn_atom_idx = torch.cdist(pts, atom_xyz).topk(1, dim=1, largest=False)[1].squeeze()
    for ix in range(len(pts)):
        res = residues[knn_res_idx[ix]]
        res_id = res.get_id()
        atom_name = atom_types[knn_atom_idx[ix]]
        res_name = knn_res_types[ix]
        if not (atom_name == 'H' and res_id in satisfied_HN) and not (atom_name == 'O' and res_id in satisfied_CO):
            hbond[ix] = computeChargeHelper(atom_name, res, res_name, pts[ix])  # Ignore atom if it is BB
    return hp, hbond


def load_point_cloud_by_file_extension(file_name, with_normal=True):
    mesh = trimesh.load(file_name, force='mesh')
    point_set = torch.tensor(mesh.vertices).float()

    if with_normal:
        vertices_normal = torch.tensor(mesh.vertex_normals).float()
        return point_set, vertices_normal
    return point_set


if __name__ == '__main__':
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB
    root = './data/cath_4.2'
    structure_cache_path = os.path.join(root, 'structure.pkl')
    surface_cache_path = os.path.join(root, 'surface_pymol_bb.lmdb')
    print(f'Loading data cache from {structure_cache_path}')
    os.makedirs(os.path.join(root, 'pdb/'), exist_ok=True)
    os.makedirs(os.path.join(root, 'surface_bb/'), exist_ok=True)

    # db_conn = lmdb.open(surface_cache_path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False, )
    # from Bio.PDB.PDBIO import PDBIO
    # from Bio.PDB import PDBParser, Selection, Select
    # import urllib.request
    # # import requests
    # with db_conn.begin(write=True, buffers=True) as txn:
    #     keys = list(txn.cursor().iternext(values=False))
    import math
    #     keys = [str(key, 'utf8') for key in keys]
    from Bio.PDB.Polypeptide import one_to_three

    bad_pdb, bad_surface = [], []
    with open(structure_cache_path, 'rb') as file:
        dataset, alphabet_set = pickle.load(file)

        template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
        for entry in tqdm(dataset):
            pdb_path = os.path.join(root, f'pdb/{entry["name"]}.pdb')
            surf_path = os.path.join(root, f'{entry["name"]}.obj')
            if os.path.exists(pdb_path):
                continue

            with open(pdb_path, 'w') as f:
                coords = entry['coords']
                for i in range(len(coords['CA'])):
                    if not math.isnan(coords['CA'][i][0]) and not math.isnan(coords['N'][i][0]) and not math.isnan(coords['C'][i][0]) and not math.isnan(coords['O'][i][0]):
                        restype = entry['seq'][i]
                        resname = one_to_three(restype)

                        xyz = coords['N'][i]
                        f.write(template.format("ATOM", 4 * i + 1, 'N', '', resname, 'A', i + 1, '', xyz[0], xyz[1], xyz[2], 1.00, 1.00, 'N', ''))
                        xyz = coords['CA'][i]
                        f.write(template.format("ATOM", 4 * i + 2, 'CA', '', resname, 'A', i + 1, '', xyz[0], xyz[1], xyz[2], 1.00, 1.00, 'C', ''))
                        xyz = coords['C'][i]
                        f.write(template.format("ATOM", 4 * i + 3, 'C', '', resname, 'A', i + 1, '', xyz[0], xyz[1], xyz[2], 1.00, 1.00, 'C', ''))
                        xyz = coords['O'][i]
                        f.write(template.format("ATOM", 4 * i + 4, 'O', '', resname, 'A', i + 1, '', xyz[0], xyz[1], xyz[2], 1.00, 1.00, 'O', ''))
    print(len(bad_pdb))
    print(bad_pdb)
    # print('bad surface', bad_surface)
