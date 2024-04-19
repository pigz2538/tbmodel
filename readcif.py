import numpy as np
import dgl
import torch
from pymatgen.core.structure import Structure
import pandas as pd
# import cugraph
# import cudf
from dgl import backend as F
from copy import deepcopy
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.core import Orbital
import scipy.constants as C

def get_orb(atom): #获取最外层轨道参数
    atom = atom.species.elements[0]
    outorb_list = atom.electronic_structure.split('.')[1:]
    aorb_dict = atom.atomic_orbitals
    outorb = dict(map(lambda x: (x[:2], int(x[-1])), outorb_list))
    orbdict = {'s':2, 'p': 6, 'd':10, 'f': 14}
    orbnum = sum(outorb.values())
    keyindex = {key: index for index, key in enumerate(aorb_dict)}.get(outorb_list[0][:2])

    if orbnum < 3 and (not outorb_list[0].startswith('1s') or not outorb_list[0].startswith('2s')):
        dictkey = list(aorb_dict.keys())[keyindex- 1] # 查找上一个轨道
        outorb_list.insert(0,  dictkey + str(orbdict[dictkey[-1]]))
        outorb = dict(map(lambda x: (x[:2], int(x[-1])), outorb_list)) 
        orbnum = sum(outorb.values())

    orb_energy = list(aorb_dict.values())[-keyindex:]
    orb_ionization_energies = atom.ionization_energies[-orbnum:]

    return outorb_list, orb_energy, orb_ionization_energies, orbnum

def equalize_ndarray(ndarr): #补0
    max_length = max(len(row) for row in ndarr)
    equalized_ndarr = np.array([np.concatenate((row, np.zeros(max_length-len(row)))) for row in ndarr])
    return np.float64(equalized_ndarr)

def crystal_to_dgl(crystal, atomic_numbers, r_neighborhood):
    a_num = atomic_numbers.size
    all_nbrs = crystal.get_neighbor_list(r_neighborhood)
    nbr_r = np.array([crystal.get_distance(i, j, jimage=0) for i,j in zip(all_nbrs[0], all_nbrs[1])])

    nbr_frame = pd.DataFrame({'index': all_nbrs[0], 'nbr_id': all_nbrs[1], 'nbr_r': nbr_r})
    nbr_frame = nbr_frame.drop_duplicates(subset=['index','nbr_id'], keep='first')
    src = np.array(nbr_frame['index'])
    dst = np.array(nbr_frame['nbr_id'])

    edge_r = np.array(nbr_frame['nbr_r'])

    graph = dgl.graph((src,dst))
    graph.ndata['species'] = torch.tensor(atomic_numbers)
    graph.edata['distance'] = torch.tensor(edge_r)
    
    lattice_frame = crystal.as_dataframe()
    radius = torch.tensor([x.specie.atomic_radius_calculated for x in crystal])
    
    mass = torch.tensor([x.specie.atomic_mass for x in crystal])
    massx = torch.sum(mass * lattice_frame['x'].to_numpy())/torch.sum(mass)
    massy = torch.sum(mass * lattice_frame['y'].to_numpy())/torch.sum(mass)
    massz = torch.sum(mass * lattice_frame['z'].to_numpy())/torch.sum(mass)
    lattice_frame['x'] = lattice_frame['x'] - massx.numpy()
    lattice_frame['y'] = lattice_frame['y'] - massy.numpy()
    lattice_frame['z'] = lattice_frame['z'] - massz.numpy()

    xa = lattice_frame['x'].to_numpy()
    ya = lattice_frame['y'].to_numpy()
    za = lattice_frame['z'].to_numpy()
    xb = torch.tensor(np.tile(xa[:,None], (1,a_num)) - np.tile(xa,(a_num,1)))
    yb = torch.tensor(np.tile(ya[:,None], (1,a_num)) - np.tile(ya,(a_num,1)))
    zb = torch.tensor(np.tile(za[:,None], (1,a_num)) - np.tile(za,(a_num,1)))

    outinfor = list(map(get_orb, crystal))
    outene = np.array([np.concatenate((x[1], x[2])) for x in outinfor], dtype=object)

    outene = torch.tensor(equalize_ndarray(outene))
    outene = outene*outene.min()/outene.max()

    graph.ndata['species'] = torch.tensor(atomic_numbers)
    graph.ndata['radius'] = radius
    coord = torch.tensor(np.stack((lattice_frame['a'], lattice_frame['b'], lattice_frame['c'], lattice_frame['x'], lattice_frame['y'], lattice_frame['z']), axis=1))
    prop = torch.stack((graph.ndata['species'], radius, mass), dim=1)
    features = torch.cat((prop, coord, xb, yb, zb, outene), dim=1)
    graph.ndata['feature'] = features[:,:60]

    return graph, outinfor

def datom_info(atom1, atom2, calorb):
    p = np.where(np.array(calorb[atom1.specie.symbol]) == 1)[0]
    q = np.where(np.array(calorb[atom2.specie.symbol]) == 1)[0]

    dti = p.repeat(q.size)
    dtj = np.tile(q, p.size)

    return np.stack((dti, dtj), axis=1), p.size, q.size

def rvector_init(dim):
    dim_num = np.sum(dim)
    rvectors = np.zeros((3 ** dim_num, 3), dtype=int)
    pk = np.arange(-1, 2, dtype=int)
    index = np.where(dim==1)[0]
    indexnum = np.arange(0, index.size)

    for i in indexnum:
        if i == 0:
            rvectors[:, index[i]]= np.repeat(pk, 3 ** (dim_num - 1))
        elif i == 1:
            rvectors[:, index[i]] = np.repeat(np.tile(pk, 3), 3 ** (dim_num - 2))
        elif i == 2:
            rvectors[:, index[i]] = np.tile(pk, 3 ** (dim_num - 1))
    return rvectors

def get_onsite_key(crystal, calorb, cell_atom_num):
    shapes = 0

    temp1 = []
    temp2 = []
    temp3 = np.array([], dtype=np.int64)
    temp4 = np.array([], dtype=np.int64)
    onsite_key = []
    for i in range(cell_atom_num):
        x = crystal.species[i]
        t1 = np.where(np.array(calorb[x.name])==1)[0]
        ti = np.repeat(t1, t1.size)
        tj = np.tile(t1, t1.size)
        tr = np.stack((ti, tj), axis=1) + i * 10
        temp1.append(t1.size) # Graph中原子序号
        temp2.append(tr) # 在位能不同轨道
        temp3 = np.append(temp3, t1)
        temp4 = np.append(temp4, np.arange(0, t1.size**2, t1.size)+np.arange(0,t1.size,1)+shapes) # 在位能同轨道
        shapes += tr.shape[0]

    temp1 = np.arange(0,cell_atom_num,1).repeat(temp1)
    
    return [temp1, temp3, temp4]

def get_orb_index(crystal, calorb, cell_atom_num, atom_num):
    orb1_indexs, orb1_indexp, orb1_indexd, orb1_indexS = [], [], [], []
    orb2_indexs, orb2_indexp, orb2_indexd, orb2_indexS = [], [], [], []
    # orb2_index = [np.zeros(cell_atom_num, 1), np.zeros((cell_atom_num, 3)), np.zeros((cell_atom_num, 5)), np.zeros(cell_atom_num, 1)]

    for i in range(atom_num):
        atom = calorb[crystal.species[i].name]

        if atom[0]:
            orb1_indexs.append(i)
            orb2_indexs.append(i)
        
        if atom[1] or atom[2] or atom[3]:
            orb1_indexp.append(i)
            orb2_indexp.append(i + np.array([0,0,0]))

        if atom[4] or atom[5] or atom[6] or atom[7] or atom[8]:
            orb1_indexd.append(i)
            orb2_indexd.append(i + np.array([0,0,0,0,0]))

        if atom[9]:
            orb1_indexS.append(i)
            orb2_indexS.append(i + i * atom_num * 9)

    orb1_index = [orb1_indexs,orb1_indexp,orb1_indexd,orb1_indexS]
    orb2_index = [orb2_indexs,orb2_indexp,orb2_indexd,orb2_indexS]

    return orb1_index, orb2_index

# def get_orb_index(crystal, calorb, cell_atom_num, atom_num):
#     orb1_indexs, orb1_indexp, orb1_indexd, orb1_indexS = [], [], [], []
#     orb2_indexs, orb2_indexp, orb2_indexd, orb2_indexS = [], [], [], []
#     # orb2_index = [np.zeros(cell_atom_num, 1), np.zeros((cell_atom_num, 3)), np.zeros((cell_atom_num, 5)), np.zeros(cell_atom_num, 1)]

#     for i in range(atom_num):
#         atom = calorb[crystal.species[i].name]

#         if atom[0]:
#             orb1_indexs.append(i)
#             orb2_indexs.append(i + i * 9)
        
#         if atom[1] or atom[2] or atom[3]:
#             orb1_indexp.append(i)
#             orb2_indexp.append(i + np.array([1,2,3]) + i * 9)

#         if atom[4] or atom[5] or atom[6] or atom[7] or atom[8]:
#             orb1_indexd.append(i)
#             orb2_indexd.append(i + np.array([4,5,6,7,8]) + i * 9)

#         if atom[9]:
#             orb1_indexS.append(i)
#             orb2_indexS.append(i + i * 9 + 9)

#     orb1_index = [orb1_indexs,orb1_indexp,orb1_indexd,orb1_indexS]
#     orb2_index = [orb2_indexs,orb2_indexp,orb2_indexd,orb2_indexS]

#     return orb1_index, orb2_index

def read_cif(cif_file, calorb):

### lattice and orbit
# s py pz px dxy dyz dz2 dxz d(x2-y2)
# 0 1  2  3  4   5   6   7    8   

    crystal = Structure.from_file(cif_file)
    crystaltrans = deepcopy(crystal)
    crystal.to_unit_cell = True

    r_neighborhood = np.power(crystal.volume, 1/3)
    images = crystal.get_neighbor_list(r_neighborhood)[2]
    dim = np.sum(np.abs(images), axis=0, dtype=int)
    dim[dim != 0] = 1
    rvectors_all = rvector_init(dim)
    rvectors = rvectors_all[:int((3 ** np.sum(dim) + 1)/2)]
    # rvectors = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,1,-1],[0,0,-1],[0,-1,0],[0,-1,-1],[0,-1,1]])

    atomic_numbers = np.array(crystal.atomic_numbers)
    cell_atom_num = len(atomic_numbers)
    cell_num = rvectors.shape[0]
    atom_num = cell_atom_num * cell_num
    onsite_key = get_onsite_key(crystal, calorb, cell_atom_num)

    hopping_info = []
    hopping_orbital = []
    hopping_index = []
    k = 0

    for ij in range(len(rvectors)):
        scaling_vector = rvectors[ij]
        cc = deepcopy(crystal)
        cc.translate_sites(range(0, cell_atom_num), vector=scaling_vector, to_unit_cell=False)
        i = 0
        for atom1 in crystal:
            j = k
            a1coord = atom1.frac_coords
            for atom2 in cc:
                orb, p, q = datom_info(atom1, atom2, calorb)
                hopping_orbital.append(orb)
                a2coord = atom2.frac_coords

                distance = atom1.distance(atom2, jimage=0)
                is_hopping = 0 if distance == 0. else 1 # 1 denotes hopping terms and 0 denotes onsite terms              
                distance = distance if distance !=0. else 1.
                l,m,n = np.dot((a2coord - a1coord),crystal.lattice.matrix) / distance

                hopping_info.append(np.array([atom1.specie.Z, atom2.specie.Z, distance, l, m, n, len(orb), is_hopping, p, q]))

                hopping_index.append([i ,j])
                j += 1
            i += 1
        k += i

        if ij > 0:
            for add_num in cc:
                crystaltrans.append(species = add_num.species, coords=add_num.coords, properties=add_num.properties, coords_are_cartesian=True)

    orb1_index, orb2_index = get_orb_index(crystaltrans, calorb, cell_atom_num, atom_num)

    atomic_numbers = np.array(crystaltrans.atomic_numbers)
    graph_s, outinfor = crystal_to_dgl(crystaltrans, atomic_numbers, r_neighborhood)
    hopping_info = np.array(hopping_info)
    hopping_orbital = hopping_orbital
    hopping_index = np.array(hopping_index)

    return graph_s, hopping_info, hopping_orbital, hopping_index, rvectors, rvectors_all, outinfor, onsite_key, orb1_index, orb2_index

