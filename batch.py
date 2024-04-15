import numpy as np
import os 
import torch
import json
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
import readcif as rd
import utils
import random

device = 'cuda:0'

def read_data(path, data_num):
    dirlist = os.listdir(path)
    subdirs = []
    for subdir in dirlist:
        if not os.path.isdir(os.path.join(path,subdir)):
            continue
        else:
            subdirs.append(subdir)

    subdirs = random.sample(subdirs, k=data_num)
    graphs = []
    labels = []
    infos = {}
    label_count = 0
    max_orbital_num = 14
    max_orbkey_num = 11

    for subdir in subdirs:
        pwd = os.path.join(path, subdir)

        with open(os.path.join(pwd, subdir + '.json'), 'r') as f:
            setjson = json.load(f)

        cif_file = os.path.join(pwd, subdir + '.cif')
        bands = np.load(os.path.join(pwd, 'bands.npy'))
        kpoints = np.load(os.path.join(pwd, 'k_points.npy'))

        label = label_count

        calorb = dict(setjson['calorb'])
        graph_s, hopping_info, hopping_orbital, hopping_index, rvectors, rvectors_all, outinfor, onsite_key, orb1_index, orb2_index = rd.read_cif(cif_file, calorb)

        graphs.append(graph_s.to(device))
        labels.append(label)
        label_count+=1

        orb_num = torch.tensor(np.array([x.shape[0] for x in hopping_orbital]))
        atom_num = graph_s.nodes().shape[0]
        init_dim = graph_s.ndata['feature'].shape[1]

        para_sk = utils.get_coefficient(rvectors, hopping_info, hopping_orbital, max_orbital_num, atom_num)

        tensor_E = torch.tensor(bands)
        tensor_K = torch.tensor(kpoints)

        tensor_eikr = utils.get_eikr(kpoints, rvectors_all)
        hopping_info = torch.tensor(hopping_info)
        hopping_index = torch.tensor(hopping_index)

        hopping_infos = hopping_info.repeat_interleave(orb_num, dim=0)
        hopping_indexs = hopping_index.repeat_interleave(orb_num, dim=0)

        rvectors = torch.tensor(rvectors)
        d = hopping_infos[:, 2].reshape([-1,1])
        coord = hopping_infos[:, 10:]
        coord = coord.reshape(-1, 2, 1)
        is_hopping = hopping_infos[:,7]
        
        calorb_num = setjson['calorb_num']
        tmp = list(map(lambda x: x*x, calorb_num))
        onsite_num = np.array(tmp)

        infos[label] = {}
        infos[label]['filename'] = setjson['filename'].replace('.py','')
        infos[label]['cell_atom_num'] = setjson['cell_atom_num']
        infos[label]['atom_num'] = atom_num
        infos[label]['d'] = torch.tensor(d).to(device)
        infos[label]['hopping_index'] = torch.tensor(hopping_indexs).to(device)
        infos[label]['hopping_info'] = torch.tensor(hopping_info).to(device)
        infos[label]['hopping_orbital'] = hopping_orbital
        infos[label]['is_hopping'] = torch.tensor(is_hopping).to(device)
        infos[label]['onsite_key'] = onsite_key
        infos[label]['onsite_num'] = onsite_num
        infos[label]['orb1_index'] = orb1_index
        infos[label]['orb2_index'] = orb2_index
        infos[label]['orb_key'] = np.array(setjson['orb_key'])
        infos[label]['orb_num'] = orb_num
        infos[label]['outinfor'] = outinfor
        infos[label]['para_sk'] = torch.tensor(para_sk).to(device)
        infos[label]['rvectors'] = torch.tensor(rvectors).to(device)
        infos[label]['rvectors_all'] = torch.tensor(rvectors_all).to(device)
        infos[label]['tensor_E'] = torch.tensor(tensor_E).to(device)
        infos[label]['tensor_eikr'] = torch.tensor(tensor_eikr).to(device)
    
    return graphs, torch.tensor(labels), infos, init_dim

class GGCNNDATASET(DGLDataset):
    """
    The dataset class used to get the dgl graph data from the raw data(crystal structures and bands data)
    """
    def __init__(self, 
                 raw_dir, 
                 save_dir,
                 data_num,
                 force_reload = False, 
                 verbose = False):
        
        self.init_dim = None
        self.data_num = data_num

        super(GGCNNDATASET, self).__init__(name="2dlayer",
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
    
    def download(self):
        pass 
    
    def process(self):
        path = self.raw_dir
        self.graphs, self.labels, self.infos, self.init_dim = read_data(path, self.data_num) 
        
    def save(self):
        graph_path = os.path.join(self.save_dir, 'graphs.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.labels})
        info_path = os.path.join(self.save_dir, 'infos.bin')
        save_info(info_path, self.infos)
        
    def has_cache(self):
        graph_path = os.path.join(self.save_dir, 'graphs.bin')
        info_path = os.path.join(self.save_dir, 'infos.bin')
        return (os.path.exists(graph_path) and os.path.exists(info_path))
    
    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_dir, 'graphs.bin'))
        infos = load_info(os.path.join(self.save_dir, 'infos.bin'))
        self.graphs = graphs.to(device)
        self.labels = label_dict['labels']
        self.infos = infos 
    
    @property
    def get_infos(self):
        return self.infos
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    def get_all(self):
        return self.graphs, self.labels, self.init_dim