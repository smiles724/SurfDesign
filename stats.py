import json
import lmdb
import pickle
import os
from tqdm import tqdm
from torch.utils.data.dataset import Subset

from src.datamodules.datasets.cath import cath_to_pdb


root = "./data/cath_4.3"
chain_set_splits_json = 'splits.json'  # cath 4.3
structure_cache_path = os.path.join(root, 'structure.pkl')
chain_set_splits_json_fullpath = os.path.join(root, chain_set_splits_json)
surf_path = os.path.join(root, 'surface_pymol_bb.lmdb')
MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB


def vertex_count():
    with open(structure_cache_path, 'rb') as file:
        dataset, alphabet_set = pickle.load(file)

    with open(chain_set_splits_json_fullpath) as f:
        dataset_splits = json.load(f)

    dataset_indices = {entry['name']: i for i, entry in enumerate(dataset)}
    db_conn = lmdb.open(surf_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )

    for i in ['test', 'train', 'validation']:
        print(len(dataset_splits[i]))
        count = []
        length = []
        tmp = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[i] if chain_name in dataset_indices])
        for entry in tmp:
            seq = entry['seq']
            with db_conn.begin() as txn:
                pts, norms, hp, hbond = pickle.loads(txn.get(entry['name'].encode()))
            length.append(len(seq))
            count.append(len(pts))
        print(i, max(count), min(count), sum([i / j for i, j in zip(count, length)]) / len(count))


def region_identity():
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    # Thresholds for RSA to determine surface vs. core
    surface_threshold = 0.25  # RSA > 0.25 is considered surface
    core_threshold = 0.15  # RSA < 0.15 is considered core

    with open(structure_cache_path, 'rb') as file:
        dataset, alphabet_set = pickle.load(file)

    identity = {}
    for entry in tqdm(dataset):
        pdb_path = os.path.join(root, f'pdb/{entry["name"]}.pdb')
        if not os.path.exists(pdb_path):
            cath_to_pdb(entry, pdb_path)

        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp='mkdssp')  # conda install -c ostrokach dssp

        results = {'loops': [], 'surface': [], 'core': []}
        for i, residue in enumerate(dssp):
            # res_id = residue[0]
            secondary_structure = residue[2]
            rsa = residue[3]

            # Categorize residues
            if secondary_structure == 'T':
                results['loops'].append(i)
            if rsa > surface_threshold:
                results['surface'].append(i)
            elif rsa < core_threshold:
                results['core'].append(i)

        identity[entry["name"]] = results

    with open('saved_dict.pkl', 'wb') as f:
        pickle.dump(identity, f)


def surface_eval(gt, pr, ):
    """   # TODO: use side-chain PDB to compute metrics instead of only backbone surface
    cd src/inside_mesh
    pip install Cpython
    python setup.py build_ext --inplace
    """
    import trimesh
    from src.metrics import compute_trimesh_chamfer, compute_iou, compute_normal_consistency
    from pymol import cmd

    cmd.load(gt + '.pdb', "A")
    cmd.load(pr + '.cif', "C")
    cmd.align("A", "C")
    cmd.save(gt + '_align.pdb', "A")

    cmd.reinitialize()
    cmd.load(gt + '_align.pdb')
    cmd.show_as('surface')
    cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1))
    cmd.save(gt + '.obj')

    cmd.reinitialize()
    cmd.load(pr + '.cif')
    cmd.show_as('surface')
    cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1))
    cmd.save(pr + '.obj')

    mesh_gt = trimesh.load(gt + '.obj')
    mesh_pr = trimesh.load(pr + '.obj')
    iou = compute_iou(mesh_gt, mesh_pr)
    chamfer_dist = compute_trimesh_chamfer(mesh_gt, mesh_pr)
    normal_consistency = compute_normal_consistency(mesh_gt, mesh_pr)

    out_dict = {"iou": iou, "chamfer_dist": chamfer_dist, "normal_consistency": normal_consistency}
    return out_dict


def calulate_tm_rmsd(pdb_1, pdb_2):
    """ https://github.com/Dapid/tmscoring """
    import tmscoring   # conda install -c conda-forge iminuit
    alignment = tmscoring.TMscoring(pdb_1, pdb_2)

    # Find the optimal alignment
    alignment.optimise()

    # Get the TM score:
    s = alignment.tmscore(**alignment.get_current_values())

    # RMSD of the protein aligned according to TM score
    r = alignment.rmsd(**alignment.get_current_values())
    return s, r


if __name__ == '__main__':
    # region_identity()
    # for pair in [['1vpa.A', 'fold_1vpa_model_0']]:   # ['3hk4.A', 'fold_3hk4_model_0']
    #     surface_eval(pair[0], pair[1])

    import os
    lm, surf = [[], []], [[], []]
    filelist = os.listdir('4.2')
    for file in filelist:
        if file.endswith('.pdb'):
            name = file[:4]
            p1 = f'4.2/{file}'
            p2 = f'4.2/fold_{name}_model_0.cif'
            p3 = f'4.2/LM-Design/fold_{name}_lm_model_0.cif'

            s, r = calulate_tm_rmsd(p1, p2)
            s_, r_ = calulate_tm_rmsd(p1, p3)
            surf[0].append(s)
            surf[1].append(r)
            lm[0].append(s_)
            lm[1].append(r_)
    print(surf[0], lm[0])
    print(surf[1], lm[1])