import numpy as np
import torch
import random
import os
from batch import GGCNNDATASET

device = 'cuda:0'

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def get_eikr(kpoints, rvectors):

    k = torch.tensor(kpoints).tile([rvectors.shape[0],1,1]).to(device)
    r = torch.tensor(rvectors).tile([kpoints.shape[0],1,1]).to(device)
    eikr = torch.exp(1j * torch.sum(k.transpose(0,1)*r, dim=2)).to(device)
        
    return eikr
        

def construct_hr(hsk, hopping_info, orb_num, cell_atom_num, rvectors):
    
    hrs = []
    i = 0

    for info, num in zip(hopping_info, orb_num):
        hrs.append(hsk[i:i+num].reshape(int(info[8]), int(info[9]), -1))
        i = i + num  

    # print(len(hrs), cell_atom_num)

    hr = []
    for i in range(0, len(hrs), cell_atom_num):
        # print([hrs[j] for j in range(i, i + atom_num)])
        tmpsk = torch.cat([hrs[j] for j in range(i, i + cell_atom_num)], 1)
        hr.append(tmpsk)
        
    HR = torch.cat(hr).to(device)
    HR = HR.reshape(rvectors.shape[0], -1, HR.shape[1])
    HR[4] =(HR[4].transpose(1,0) + HR[4])/2

    HR5 = HR[3].clone().transpose(1,0).unsqueeze(0)
    HR6 = HR[2].clone().transpose(1,0).unsqueeze(0)
    HR7 = HR[1].clone().transpose(1,0).unsqueeze(0)
    HR8 = HR[0].clone().transpose(1,0).unsqueeze(0)  

    HR = torch.cat([HR, HR5, HR6, HR7, HR8], dim=0)
    return HR

def compute_bands(HR, eikr):

    hr = HR.tile([eikr.shape[0], 1, 1, 1])
    er = torch.unsqueeze(torch.unsqueeze(eikr, 2), 3)
    HK = torch.sum(hr*er, dim=1)
    w, v = torch.linalg.eigh(HK)

    return w

def get_data(raw_dir, save_dir, data_num, force_reload=False):
    data = GGCNNDATASET(raw_dir, save_dir, data_num, force_reload)
    infos = data.infos
    return data, infos 
 
def get_coefficient(rvectors, hopping_info, hopping_orbital, max_orbital_num, atom_num):
    def switch_case(case, default):
        return lambda *args: case.get(args[0], default)(*args[1:])
    sqrt3 = np.sqrt(3)
    # Vsss Vsps Vpps Vppp Vsds Vpds Vpdp Vdds Vddp Vddd VSSs VsSs VSps VSds
    # [0   0    0    0    0    0    0    0    0    0    0    0    0    0   ]
    sk_cases = {
        (0, 0): lambda l, m, n: [1], # s s
        (0, 1): lambda l, m, n: [0, l, 0, 0], # s px
        (0, 2): lambda l, m, n: [0, m, 0, 0], # s py
        (0, 3): lambda l, m, n: [0, n, 0, 0], # s pz
        (0, 4): lambda l, m, n: [0, 0, 0, 0, sqrt3*l*m, 0, 0], # s dxy
        (0, 5): lambda l, m, n: [0, 0, 0, 0, sqrt3*m*n, 0, 0], # s dyz
        (0, 6): lambda l, m, n: [0, 0, 0, 0, sqrt3*l*n, 0, 0], # s dxz
        (0, 7): lambda l, m, n: [0, 0, 0, 0, 0.5*sqrt3*(l**2-m**2), 0, 0], # s d(x2-y2)
        (0, 8): lambda l, m, n: [0, 0, 0, 0, n**2-0.5*(l**2+m**2), 0, 0], # s dz2
        (0, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # s S
        (1, 0): lambda l, m, n: [0, -l, 0, 0], # px s
        (1, 1): lambda l, m, n: [0, 0, l**2, (1-l**2)], # px px
        (1, 2): lambda l, m, n: [0, 0, l*m, -l*m], # px py
        (1, 3): lambda l, m, n: [0, 0, l*n, -l*n], # px pz
        (1, 4): lambda l, m, n: [0, 0, 0, 0, 0, sqrt3*(l**2)*m, m*(1-2*(l**2))], # px dxy
        (1, 5): lambda l, m, n: [0, 0, 0, 0, 0, sqrt3*l*m*n, -2*l*m*n], # px dyz
        (1, 6): lambda l, m, n: [0, 0, 0, 0, 0, sqrt3*(l**2)*n, n*(1-2*(l**2))], # px dxz
        (1, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0.5*sqrt3*l*(l**2-m**2), l*(1-l**2+m**2)], # px d(x2-y2)
        (1, 8): lambda l, m, n: [0, 0, 0, 0, 0, l*(n**2-0.5*(l**2-m**2)), -sqrt3*l*(n**2)], # px dz2
        (1, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -l], # px S
        (2, 0): lambda l, m, n: [0, -m, 0, 0], # py s
        (2, 1): lambda l, m, n: [0, 0, m*l, -m*l], # py px
        (2, 2): lambda l, m, n: [0, 0, m**2, (1-m**2)], # py py
        (2, 3): lambda l, m, n: [0, 0, m*n, -m*n], # py pz
        (2, 4): lambda l, m, n: [0, 0, 0, 0, 0, sqrt3*(m**2)*l, l*(1-2*(m**2))], # py dxy
        (2, 5): lambda l, m, n: [0, 0, 0, 0, 0, sqrt3*(m**2)*n, n*(1-2*(m**2))], ## pz dyz
        (3, 6): lambda l, m, n: [0, 0, 0, 0, 0, l*(n**2)*sqrt3, -l*(1-2*(n**2))], # pz dxz
        (3, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0.5*sqrt3*n*(l**2-m**2), -n*(l**2-m**2)], # pz d(x2-y2)
        (3, 8): lambda l, m, n: [0, 0, 0, 0, 0, n*(n**2-0.5*(l**2+m**2)), sqrt3*n*(l**2+m**2)], # pz dz2
        (3, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -n], # pz S
        (4, 0): lambda l, m, n: [0, 0, 0, 0, sqrt3*l*m, 0, 0], # dxy s
        (4, 1): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*(l**2)*m, -m*(1-2*(l**2))], # dxy px
        (4, 2): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*(m**2)*l, -l*(1-2*(m**2))], # dxy py
        (4, 3): lambda l, m, n: [0, 0, 0, 0, 0, -l*m*n*sqrt3, l*m*n*2], # dxy pz
        (4, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, (l**2)*(m**2)*3, (l**2+m**2)-(l**2)*(m**2)*4, (l**2)*(m**2)+(n**2)], # dxy dxy
        (4, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, l*(m**2)*n*3, l*n-l*(m**2)*n*4, l*(m**2)*n-l*n], # dxy dyz
        (4, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, n*(l**2)*m*3, m*n-n*(l**2)*m*4, n*(l**2)*m-m*n], # dxy dxz
        (4, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*l*m*(l**2-m**2)*3, -0.5*l*m*(l**2-m**2)*4, 0.5*l*m*(l**2-m**2)], # dxy d(x2-y2)
        (4, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(l*m*(n**2-0.5*(l**2+m**2))), -sqrt3*2*l*m*n**2, sqrt3*0.5*l*m*(1+n**2)], # dxy dz2
        (4, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*l*m], # dxy S
        (5, 0): lambda l, m, n: [0, 0, 0, 0, sqrt3*m*n, 0, 0], # dyz s
        (5, 1): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*l*m*n, 2*l*m*n], # dyz px
        (5, 2): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*(m**2)*n, -n*(1-2*(m**2))], # dyz py
        (5, 3): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*(n**2)*m, -m*(1-2*(n**2))], # dyz pz
        (5, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, l*(m**2)*n*3, l*n-l*(m**2)*n*4, l*(m**2)*n-l*n], # dyz dxy
        (5, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, (m**2)*(n**2)*3, (m**2+n**2)-(m**2)*(n**2)*4, (m**2)*(n**2)+(l**2)], # dyz dyz
        (5, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, m*(n**2)*l*3, m*l-m*(n**2)*l*4, m*(n**2)*l-m*l], # dyz dxz
        (5, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*m*n*(l**2-m**2)*3, -0.5*m*n*(l**2-m**2)*4-2, 0.5*m*n*(l**2-m**2)+2], # dyz d(x2-y2)
        (5, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(m*n*(n**2-0.5*(l**2+m**2))), +sqrt3*m*n*(l**2+m**2-n**2), sqrt3*0.5*m*n*(l**2+m**2)], # dyz dz2
        (5, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*m*n], # dyz S
        (6, 0): lambda l, m, n: [0, 0, 0, 0, sqrt3*l*n, 0, 0], # dxz s
        (6, 1): lambda l, m, n: [0, 0, 0, 0, 0, -sqrt3*(l**2)*n, -n*(1-2*(l**2))], # dxz px
        (6, 2): lambda l, m, n: [0, 0, 0, 0, 0, -l*m*n*sqrt3, l*m*n*2], # dxz py
        (6, 3): lambda l, m, n: [0, 0, 0, 0, 0, -l*(n**2)*sqrt3, l*(1-2*(n**2))], # dxz pz
        (6, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, n*(l**2)*m*3, m*n-n*(l**2)*m*4, n*(l**2)*m-m*n], # dxz dxy
        (6, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, m*(n**2)*l*3, m*l-m*(n**2)*l*4, m*(n**2)*l-m*l], # dxz dyz
        (6, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, (n**2)*(l**2)*3, (n**2+l**2)-(n**2)*(l**2)*4, (n**2)*(l**2)+(m**2)], # dxz dxz
        (6, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*n*l*(l**2-m**2)*3, -0.5*n*l*(l**2-m**2)*4+2, 0.5*n*l*(n**2-l**2)-2], # dxz d(x2-y2)
        (6, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(n*l*(n**2-0.5*(l**2+m**2))), sqrt3*n*l*(l**2+m**2-n**2), sqrt3*0.5*n*l*(l**2+m**2)], # dxz dz2
        (6, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*l*n], # dxz S
        (7, 0): lambda l, m, n: [0, 0, 0, 0, 0.5*sqrt3*(l**2-m**2), 0, 0], # d(x2-y2) s
        (7, 1): lambda l, m, n: [0, 0, 0, 0, 0, -0.5*sqrt3*l*(l**2-m**2), -l*(1-l**2+m**2)], # d(x2-y2) px
        (7, 2): lambda l, m, n: [0, 0, 0, 0, 0, -0.5*sqrt3*m*(l**2-m**2), m*(1+l**2-m**2)], # d(x2-y2) py
        (7, 3): lambda l, m, n: [0, 0, 0, 0, 0, -0.5*sqrt3*n*(l**2-m**2), n*(l**2-m**2)], # d(x2-y2) pz
        (7, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*l*m*(l**2-m**2)*3, -0.5*l*m*(l**2-m**2)*4, 0.5*l*m*(l**2-m**2)], # d(x2-y2) dxy
        (7, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*m*n*(l**2-m**2)*3, -0.5*m*n*(l**2-m**2)*4-2, 0.5*m*n*(l**2-m**2)+2], # d(x2-y2) dyz
        (7, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.5*n*l*(l**2-m**2)*3, -0.5*n*l*(l**2-m**2)*4+2, 0.5*n*l*(n**2-l**2)-2], # d(x2-y2) dxz
        (7, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.25*((l**2-m**2)**2)*3, (l**2+m**2)-0.25*((l**2-m**2)**2)*4, 0.25*((l**2-m**2)**2)*4+(n**2)], # d(x2-y2) d(x2-y2)
        (7, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*0.25*(l**2-m**2)*((n** 2)*2-(l**2+m**2)), -sqrt3*0.25*(l**2-m**2)*((n**2)*4), sqrt3*0.25*(l**2-m**2)*((n**2)+1)], # d(x2-y2) dz2
        (7, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3/2*(l**2-m**2)], # d(x2-y2) S
        (8, 0): lambda l, m, n: [0, 0, 0, 0, n**2-0.5*(l**2+m**2), 0, 0], # dz2 s
        (8, 1): lambda l, m, n: [0, 0, 0, 0, 0, -l*(n**2-0.5*(l**2-m**2)), sqrt3*l*(n**2)], # dz2 px
        (8, 2): lambda l, m, n: [0, 0, 0, 0, 0, -m*(n**2-0.5*(l**2+m**2)), -sqrt3*m*(n**2)], # dz2 py
        (8, 3): lambda l, m, n: [0, 0, 0, 0, 0, -n*(n**2-0.5*(l**2+m**2)), -sqrt3*n*(l**2+m**2)], # dz2 pz
        (8, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(l*m*(n**2-0.5*(l**2+m**2))), -sqrt3*2*l*m*n**2, sqrt3*0.5*l*m*(1+n**2)], # dz2 dxy
        (8, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(m*n*(n**2-0.5*(l**2+m**2))), +sqrt3*m*n*(l**2+m**2-n**2), sqrt3*0.5*m*n*(l**2+m**2)], # dz2 dyz
        (8, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*(n*l*(n**2-0.5*(l**2+m**2))), sqrt3*n*l*(l**2+m**2-n**2), sqrt3*0.5*n*l*(l**2+m**2)], # dz2 dxz
        (8, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, sqrt3*0.25*(l**2-m**2)*((n** 2)*2-(l**2+m**2)), -sqrt3*0.25*(l**2-m**2)*((n**2)*4), sqrt3*0.25*(l**2-m**2)*((n**2)+1)], # dz2 d(x2-y2)
        (8, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0.25*((l**2)+(m**2)-2*(n**2))**2, 3*((l**2)+(m**2))*(n**2), 0.75*((l**2)+(m**2))**2], # dz2 dz2
        (8, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ((n**2)-0.5*((l**2)+(m**2)))], # dz2 S
        (9, 0): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # S s
        (9, 1): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l], # S px
        (9, 2): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m], # S py
        (9, 3): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n], # S pz
        (9, 4): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*l*m], # S dxy
        (9, 5): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*m*n], # S dyz
        (9, 6): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3*l*n], # S dxz
        (9, 7): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sqrt3/2*(l**2-m**2)], # S d(x2-y2)
        (9, 8): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ((n**2)-0.5*((l**2)+(m**2)))], # S dz2
        (9, 9): lambda l, m, n: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # S s
    }
    switch = switch_case(sk_cases, lambda l, m, n: [0])
    
    para_sk = []
    for orbital, info in zip(hopping_orbital, hopping_info):
        l, m, n = info[3:6]
        for o in orbital:
            sk = np.zeros(max_orbital_num)
            tmp_o = np.array(switch(tuple(o), l, m, n))
            sk[:tmp_o.size] = tmp_o
            para_sk.append(sk)

    return torch.tensor(para_sk)

def batch_index(train_dataloader, infos, batch_size):
    para_sk, hopping_index, hopping_info, d, is_hopping, onsite_key, cell_atom_num, onsite_num, orb1_index, orb2_index, orb_num, rvectors, rvectors_all, tensor_E, tensor_eikr, orb_key, filename = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for graphs, labels in train_dataloader:
        label = labels.numpy()
        sum_atom, hopping_index_batch, hopping_info_batch, para_sk_batch, is_hopping_batch, d_batch, cell_atom_num_batch, onsite_num_batch, orb1_index_batch, orb2_index_batch, orb_num_batch, rvectors_batch, rvectors_all_batch, tensor_E_batch, tensor_eikr_batch, filename_batch = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        orb_key_batch = np.zeros(14)

        ok1, ok2, ok3 = [], [], []

        obm1, obm2, obm3, obm4 = [], [], [], []
        obn1, obn2, obn3, obn4 = [], [], [], []

        for i in label:
            sum_atom.append(infos[i]['atom_num'])
            cell_atom_num_batch.append(infos[i]['cell_atom_num'])
            add_num = sum(sum_atom[0:]) - infos[i]['atom_num']
            hopping_index_batch.append(infos[i]['hopping_index'] + add_num)
            hopping_info_batch.append(infos[i]['hopping_info'])
            para_sk_batch.append(infos[i]['para_sk'])
            is_hopping_batch.append(infos[i]['is_hopping'])
            d_batch.append(infos[i]['d'])
            onsite_num_batch.append(infos[i]['onsite_num'])
            ok1.append(infos[i]['onsite_key'][0] + add_num)
            ok2.append(infos[i]['onsite_key'][1])
            ok3.append(infos[i]['onsite_key'][2] + sum(sum(onsite_num_batch[0:]) - infos[i]['onsite_num']))

            obm1 += list((np.array(infos[i]['orb1_index'][0]) + add_num))
            obm2 += list((np.array(infos[i]['orb1_index'][1]) + add_num))
            obm3 += list((np.array(infos[i]['orb1_index'][2]) + add_num))
            obm4 += list((np.array(infos[i]['orb1_index'][3]) + add_num))

            # obn1 += list((np.array(infos[i]['orb2_index'][0]) + add_num * 10))
            # obn2 += list((np.array(infos[i]['orb2_index'][1]) + add_num * 10))
            # obn3 += list((np.array(infos[i]['orb2_index'][2]) + add_num * 10))
            # obn4 += list((np.array(infos[i]['orb2_index'][3]) + add_num * 10))


            obn1 += list((np.array(infos[i]['orb2_index'][0]) + add_num))

            orb_num_batch.append(infos[i]['orb_num'])

            rvectors_batch.append(infos[i]['rvectors'])
            rvectors_all_batch.append(infos[i]['rvectors_all'])
            tensor_eikr_batch.append(infos[i]['tensor_eikr'])
            tensor_E_batch.append(infos[i]['tensor_E'])
            filename_batch.append(infos[i]['filename'])

            orb_key_batch += infos[i]['orb_key']

        for i, j in zip(label, range(len(label))):
            sum_cell = sum(sum_atom[0:j]) 
            obn2 += list((np.array(infos[i]['orb2_index'][1]) + np.array([1,2,3]) * sum(sum_atom) + sum_cell))
            obn3 += list((np.array(infos[i]['orb2_index'][2]) + np.array([4,5,6,7,8])* sum(sum_atom) + sum_cell))
            obn4 += list((np.array(infos[i]['orb2_index'][3]) + sum(sum_atom) * 9 + sum_cell))


        hopping_index.append(torch.cat(hopping_index_batch))
        hopping_info.append(torch.cat(hopping_info_batch))
        is_hopping.append(torch.cat(is_hopping_batch))
        d.append(torch.cat(d_batch))
        para_sk.append(torch.cat(para_sk_batch))
        orb_num.append(torch.cat(orb_num_batch))
        rvectors.append(torch.stack(rvectors_batch, dim=0))
        rvectors_all.append(torch.stack(rvectors_all_batch, dim=0))

        tensor_eikr.append(torch.stack(tensor_eikr_batch, dim=0))
        tensor_E.append(torch.stack(tensor_E_batch, dim=0))

        onsite_key.append([np.concatenate(ok1), np.concatenate(ok2), np.concatenate(ok3)])

        cell_atom_num.append(sum(cell_atom_num_batch))
        onsite_num.append(np.concatenate(onsite_num_batch))

        orb1_index.append([obm1, obm2, obm3, obm4])
        orb2_index.append([obn1, obn2, obn3, obn4])

        orb_key.append(np.sign(orb_key_batch))

        filename.append(filename_batch)

    return para_sk, hopping_index, hopping_info, d, is_hopping, onsite_key, cell_atom_num, onsite_num, orb1_index, orb2_index, orb_num, rvectors, rvectors_all, tensor_E, tensor_eikr, orb_key, filename
