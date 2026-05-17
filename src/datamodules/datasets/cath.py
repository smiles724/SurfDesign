import json
import math
import os
import urllib.request
import pickle
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Dict, List, Sequence, Tuple

import esm
import lmdb
import subprocess
import numpy as np
import torch
import torch_cluster
import trimesh
from Bio.PDB import PDBParser, Selection, Select
from Bio.PDB.Polypeptide import one_to_three
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.PDBIO import PDBIO

from torch.nn import functional as F
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.dataset import Subset
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html -i https://pypi.tuna.tsinghua.edu.cn/simple
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import KNNGraph  # conda install pyg -c pyg
from tqdm import tqdm

from src import utils
from .chem.computeCharges import computeChargeHelper, computeSatisfied_CO_HN
from .chem.computeHydro import computeHydrophobicity
from .chem.constants import BBHeavyAtom
from .data_utils import Alphabet

Tensor, tensor = torch.LongTensor, torch.FloatTensor
log = utils.get_logger(__name__)
MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB


class NonHetAndChainSelect(Select):
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

    def accept_chain(self, chain):
        return chain.get_id() in self.chain_letters


def load_point_cloud_by_file_extension(file_name, with_normal=True):
    mesh = trimesh.load(file_name, force='mesh')
    point_set = torch.tensor(mesh.vertices).float()

    if with_normal:
        vertices_normal = torch.tensor(mesh.vertex_normals).float()
        return point_set, vertices_normal
    return point_set


def read_msms(vertfile):
    """
    https://github.com/vsomnath/holoprot/blob/a22794ebddf563879221d17384e8ba2f36c09f1f/holoprot/utils/surface.py
    """
    pts, norms = [], []
    with open(vertfile) as f:
        meshdata = (f.read().rstrip()).split("\n")
        header = meshdata[2].split()
        num_vert = int(header[0])

        for i in range(3, len(meshdata)):  # skip header
            fields = meshdata[i].split()
            pts.append(tensor([float(fields[0]), float(fields[1]), float(fields[2])]))
            norms.append(tensor([float(fields[3]), float(fields[4]), float(fields[5])]))
            num_vert -= 1
        assert num_vert == 0
    return torch.stack(pts, dim=0), torch.stack(norms, dim=0)


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


def cath_to_pdb(entry, pdb_path):
    template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
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


def surface_smooth(vertices):
    """ smooth function in SurfPro, https://github.com/JocelynSong/SurfPro """
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(vertices.numpy())
    distances, indices = nbrs.kneighbors(vertices.numpy())  # [N, 8]
    distances = np.square(distances)
    d = np.max(distances, axis=1, keepdims=True)  # [N, 8]
    probs = np.exp(-distances / d)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    new_vertices = []
    for prob, index in zip(probs, indices):
        neighbors = torch.stack([vertices[ind] for ind in index], dim=0)  # [8, 3]
        vertex = torch.sum(neighbors * torch.from_numpy(prob.reshape(-1, 1)), dim=0)
        new_vertices.append(vertex)
    return torch.stack(new_vertices, dim=0)


def CATH(root=".data", chain_set_jsonl='chain_set.jsonl', chain_set_splits_json='chain_set_splits.json', split=("train", "validation", "test"), max_length=500,
         alphabet='ACDEFGHIKLMNPQRSTVWY', transforms: Callable = (None, None), bb_surface=True, surf_mode='pymol', verbose=False):
    chain_set_jsonl_fullpath = os.path.join(root, chain_set_jsonl)
    chain_set_splits_json_fullpath = os.path.join(root, chain_set_splits_json)
    structure_cache_path = os.path.join(root, 'structure.pkl')
    surface_cache_path = os.path.join(root, f'surface_{surf_mode}{"_bb" if bb_surface else ""}.lmdb')

    # 1) load the dataset
    split = ['validation' if s == 'valid' else s for s in split]
    if not os.path.exists(structure_cache_path):
        alphabet_set = set([a for a in alphabet])
        discard_count = {'bad_chars': 0, 'too_long': 0, }
        dataset: List[Dict] = []

        with open(chain_set_jsonl_fullpath) as f:
            # NOTE: dataset is a list of mapping
            #   name: str
            #   seq: str. sequence of amino acids
            #   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}
            print(chain_set_jsonl_fullpath)
            for i, line in enumerate(f.readlines()):
                if 'cath_4.3' in chain_set_jsonl_fullpath:
                    entry = json.loads(r'{}'.format(line))
                else:
                    entry = json.loads(line)
                seq = entry['seq']
                bad_chars = set([s for s in seq]).difference(alphabet_set)    # Check if in alphabet
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        mask = ~np.isnan(entry['coords']['CA'])[:, 0]   # filter nan coords
                        entry['seq'] = ''.join(np.array(list(entry['seq']))[mask])
                        for key, val in entry['coords'].items():
                            entry['coords'][key] = np.asarray(val, dtype=np.float32)[mask]   # Convert raw coords to np arrays
                        dataset.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    discard_count['bad_chars'] += 1

                if verbose and (i + 1) % 100000 == 0:
                    print('{} entries ({} loaded)'.format(len(dataset), i + 1))
            total_size = i

        dataset = SequenceWrapper(dataset)
        with open(structure_cache_path, 'wb') as file:
            pickle.dump([dataset, alphabet_set], file)
        log.info(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')
    else:
        log.info(f'Loading data cache from {structure_cache_path}')
        with open(structure_cache_path, 'rb') as file:
            dataset, alphabet_set = pickle.load(file)

    # preprocess surface
    if not os.path.exists(surface_cache_path) or os.path.getsize(surface_cache_path) < 1e9:  # 1GB size
        os.makedirs(os.path.join(root, f'pdb{"_bb" if bb_surface else ""}/'), exist_ok=True)
        os.makedirs(os.path.join(root, f'surface{"_bb" if bb_surface else ""}/'), exist_ok=True)

        if surf_mode == 'pymol':
            from pymol import cmd
            import pymol2
            import pymol.invocation

            pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag of PyMol
            pymol2.SingletonPyMOL().start()

        db_conn = lmdb.open(surface_cache_path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False, )
        for entry in tqdm(dataset):   # TODO: cath 4.2 & 4.3 have intersections, and PDB/OBJ can be reused
            pdb_path = os.path.join(root, f'pdb{"_bb" if bb_surface else ""}/{entry["name"]}.pdb')
            if not os.path.exists(pdb_path):
                if bb_surface:
                    cath_to_pdb(entry, pdb_path)
                else:    # use the complete PDB instead of backbone surface
                    pdb_name, chain_name = entry["name"].split('.')
                    pdb_path_new = os.path.join(root, f'pdb/{pdb_name}.pdb')
                    urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_name}.pdb', pdb_path_new)   # download PDB

                    parser = PDBParser()
                    io = PDBIO()
                    structure = parser.get_structure(pdb_name, pdb_path_new)
                    io.set_structure(structure)
                    io.save(pdb_path, select=NonHetAndChainSelect(chain_name))   # select and save chain

            if surf_mode == 'biopython':  # install MSMS before calling biopython
                parser = PDBParser()
                model = parser.get_structure(entry["name"], pdb_path)[0]
                pts, norms = tensor(get_surface(model)), None   # biopython do not return norms

            elif surf_mode == 'msms':
                xyzr_path = os.path.join(root, f'surface/{entry["name"]}.xyzr')
                cmd_xyzr = ' '.join(['pdb_to_xyzr', pdb_path, '>', xyzr_path])
                subprocess.run(cmd_xyzr, shell=True)  # execute through the shell
                cmd_msms = ' '.join(['msms', '-if', xyzr_path, '-of', os.path.join(root, f'surface/{entry["name"]}')])
                subprocess.run(cmd_msms, shell=True)  # TODO: segmentation fault for '2j6v.A'
                pts, norms = read_msms(os.path.join(root, f'surface/{entry["name"]}.vert'))

            elif surf_mode == 'pymol':
                surf_path = os.path.join(root, f'surface{"_bb" if bb_surface else ""}/{entry["name"]}.obj')
                if not os.path.exists(surf_path) or os.path.getsize(surf_path) == 0:
                    cmd.reinitialize()
                    cmd.load(pdb_path)
                    cmd.show_as('surface')  # pymol has more points than MSMS  # todo: SAVE ERROR for '1a02.J' but work in PC
                    # adjust the coordinates of vertices, https://pymolwiki.org/index.php/Model_Space_and_Camera_Space
                    cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 300, 1))
                    cmd.save(surf_path)
                pts, norms = load_point_cloud_by_file_extension(surf_path)
            else:
                raise ValueError('Only support Pymol, Biopython, and MSMS.')

            hp, hbond = computer_property(pdb_path, pts, surf_mode)
            with db_conn.begin(write=True, buffers=True) as txn:   # do not nest surface extraction inside lmdb save
                txn.put(entry['name'].encode('utf-8'), pickle.dumps([pts, norms, hp, hbond]))

        db_conn.close()

    # 2) split the dataset
    dataset_indices = {entry['name']: i for i, entry in enumerate(dataset)}
    with open(chain_set_splits_json_fullpath) as f:
        dataset_splits = json.load(f)

    # compatible with cath data
    dataset_splits = [Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key] if chain_name in dataset_indices]) for key in split]
    sizes = [f'{split[i]}: {len(dataset_splits[i])}' for i in range(len(split))]
    log.info(f'Size. {", ".join(sizes)}')
    if len(dataset_splits) == 1:
        dataset_splits = dataset_splits[0]
    return dataset_splits, alphabet_set


class CoordBatchConverter(esm.data.BatchConverter):
    def __init__(self, alphabet, coord_pad_inf=False, coord_nan_to_zero=True, ):
        super().__init__(alphabet)
        self.coord_pad_inf = coord_pad_inf
        self.coord_nan_to_zero = coord_nan_to_zero

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x n_atoms x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        if self.coord_pad_inf:
            # pad beginning and end of each protein due to legacy reasons
            coords = [F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.nan) for cd, _ in coords_and_confidence]
            confidence = [F.pad(torch.tensor(cf), (1, 1), value=-1.) for _, cf in coords_and_confidence]
        else:
            coords = [torch.tensor(cd) for cd, _ in coords_and_confidence]
            confidence = [torch.tensor(cf) for _, cf in coords_and_confidence]

        coords = self.collate_dense_tensors(coords, pad_v=np.nan)     # pad as nan
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum([-2, -1]))     # compute mask based on nan values
        confidence = confidence * coord_mask + (-1.) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.

        return coords, confidence, strs, tokens, lengths, coord_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x n_atoms x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


class Featurizer(object):
    def __init__(self, surf_path, alphabet: Alphabet, coord_nan_to_zero=True, atoms=('N', 'CA', 'C', 'O'), num_neighbors=4, radius=None, smooth=False):
        self.db_conn = None
        self.surf_path = surf_path
        self.alphabet = alphabet
        self.atoms = atoms  # atom order matters!
        self.batcher = CoordBatchConverter(alphabet=alphabet, coord_pad_inf=alphabet.add_special_tokens, coord_nan_to_zero=coord_nan_to_zero)

        self.smooth = smooth
        self.radius = radius
        self.kg = KNNGraph(num_neighbors, num_workers=4)

    def get_surface(self, name, smooth):
        if self.db_conn is None:
            self.db_conn = lmdb.open(self.surf_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
        with self.db_conn.begin() as txn:
            pts, norms, hp, hbond = pickle.loads(txn.get(name.encode()))
        if smooth:
            pts = surface_smooth(pts)
        return pts, norms, hp, hbond

    def __call__(self, raw_batch: dict, dist_threshold: float = 4.0, ):  # 4.0 is enough to include all backbone residues
        seqs, coords, names, prot_surfs = [], [], [], []
        for entry in raw_batch:
            # [L, 3] x 4 -> [L, 4, 3]
            if isinstance(entry['coords'], dict):
                coords.append(np.stack([entry['coords'][atom] for atom in self.atoms], 1))
            else:
                coords.append(entry['coords'])
            seqs.append(entry['seq'])
            names.append(entry['name'])
            pts, norms, hp, hbond = self.get_surface(entry['name'], self.smooth)

            # connections are built based on radius rather than KNN
            Ca_coord = tensor(coords[-1][:, 1, :])  # Ca is adequate to assign edges
            Ca_coord = torch.nan_to_num(Ca_coord, 1e10)  # torch_cluster corrupts for nan values
            edge_index = torch_cluster.radius(Ca_coord, pts, dist_threshold)  # surface -> backbone

            p_s_g = HeteroData(protein={'pos': Ca_coord}, surface={'pos': pts, 'norm': norms, 'hp': hp, 'hbond': hbond}, )
            p_s_g['surface', 'inter', 'protein'].edge_index = edge_index
            prot_surfs.append(p_s_g)

        if len(prot_surfs) > 0:
            prot_surfs = Batch.from_data_list(prot_surfs)
            self.kg(prot_surfs['surface'])

        coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(coords_list=coords, confidence_list=None, seq_list=seqs)
        batch = {'coords': coords, 'tokens': tokens, 'confidence': confidence, 'coord_mask': coord_mask, 'lengths': lengths, 'seqs': seqs, 'names': names, 'prot_surfs': prot_surfs}
        return batch
