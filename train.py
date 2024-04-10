import numpy as np
import torch
import torch.nn as nn
import random
import utils
import numpy as np
import time
import json
import os
from batch import GGCNNDATASET
from model import WHOLEMODEL
from dgl.dataloading import GraphDataLoader
from dgl import batch
import warnings
 
warnings.filterwarnings('ignore')  # 忽略所有警告

device = 'cuda:0'

seed = 1 # seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed) # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed) # numpy产生的随机数一致
random.seed(seed) # python产生的随机数一致

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float64)

def train(dist_path):

    train_data_path = os.path.join(dist_path,'datas/train_data')
    test_data_path = os.path.join(dist_path,'datas/test_data')
    config_json_file = os.path.join(dist_path, 'datas/config.json')
    if not os.path.exists(os.path.join(dist_path, 'results')):
        os.makedirs(os.path.join(dist_path, 'results'), exist_ok=True)
    latest_point_path = os.path.join(dist_path, 'results/test_latest.pkl')

    with open(config_json_file, 'r', encoding='utf-8') as f:
        config_para = json.load(f)

    # configure hyper parameters
    batch_size     = config_para['batch_size']
    num_epoch      = config_para['num_epoch']
    lr_radio_init  = config_para['lr_radio_init']
    lr_factor      = config_para['lr_factor']
    lr_patience    = config_para['lr_patience']
    lr_verbose     = config_para['lr_verbose']
    lr_threshold   = config_para['lr_threshold']
    lr_eps         = config_para['lr_eps']
    min_lr         = config_para['min_lr']
    cooldown       = config_para['cooldown']
    is_sch         = config_para['is_sch']
    save_frequncy  = config_para['save_frequncy']

    is_L1          = config_para['is_L1']
    is_L2          = config_para['is_L2']
    L1_radio       = config_para['L1_radio']
    L2_radio       = config_para['L2_radio']

    reset_all        = config_para['reset_all']
    reset_model      = config_para['reset_model']
    reset_model_path = config_para['model_path']
    reset_opt        = config_para['reset_opt']
    reset_sch        = config_para['reset_sch']

    # configure trainingset path
    trainset_rawdata_path = os.path.join(train_data_path, 'raw')
    trainset_dgldata_path = os.path.join(train_data_path, 'dgl')

    # configure trainingset path
    testset_rawdata_path = os.path.join(test_data_path, 'raw')
    testset_dgldata_path = os.path.join(test_data_path, 'dgl')

    # configure network structure
    gnn_dim_list           = config_para['gnn_dim_list']
    gnn_head_list          = config_para['gnn_head_list']
    onsite_dim_list        = config_para['onsite_dim_list']
    orb_dim_list           = config_para['orb_dim_list']
    hopping_dim_list1      = config_para['hopping_dim_list1']
    hopping_dim_list2      = config_para['hopping_dim_list2']
    expander_bessel_dim    = config_para['expander_bessel_dim']
    expander_bessel_cutoff = config_para['expander_bessel_cutoff']
    atom_num               = config_para['atom_num']
    is_orb                 = config_para['is_orb']

    utils.seed_torch(seed = 24)

    trainset, traininfos = utils.get_data(
                                            raw_dir = trainset_rawdata_path, 
                                            save_dir = trainset_dgldata_path, 
                                            force_reload = True,
                                            )

    traingraphs, trainlabels, init_dim = trainset.get_all()
    train_num = len(traingraphs)
    traingraphs = batch(traingraphs)
    traingraphs = traingraphs.to(device)
    train_dataloader = GraphDataLoader(trainset, batch_size = batch_size, drop_last = False, shuffle = False)

    testset, testinfos = utils.get_data(
                                        raw_dir = testset_rawdata_path, 
                                        save_dir = testset_dgldata_path, 
                                        force_reload = True,
                                        )

    testgraphs, testlabels, init_dim = testset.get_all()
    test_num = len(testgraphs)
    testgraphs = batch(testgraphs)
    testgraphs = testgraphs.to(device)
    test_dataloader = GraphDataLoader(testset, batch_size = batch_size, drop_last = False, shuffle = False)

    model = WHOLEMODEL(
                        gnn_dim_list = gnn_dim_list,
                        gnn_head_list = gnn_head_list,
                        orb_dim_list = orb_dim_list,
                        onsite_dim_list = onsite_dim_list,
                        hopping_dim_list1 = hopping_dim_list1,
                        hopping_dim_list2 = hopping_dim_list2,
                        expander_bessel_dim = expander_bessel_dim,
                        expander_bessel_cutoff = expander_bessel_cutoff,
                        atom_num=atom_num*batch_size,
                        is_orb = is_orb
                        )

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr_radio_init, eps=lr_eps)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=lr_factor, patience=lr_patience, verbose=lr_verbose, threshold=lr_threshold, threshold_mode='rel', cooldown=cooldown, min_lr=min_lr, eps=lr_eps)

    criterion = nn.SmoothL1Loss()
    loss_per_epoch = np.zeros(train_num)  
    losses = np.zeros(num_epoch)
    test_losses = np.zeros(num_epoch)

    if os.path.exists(latest_point_path) and not reset_all:
        checkpoint = torch.load(latest_point_path)
        if not reset_model:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = torch.load(reset_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        if not reset_opt:
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if not reset_sch:
            sch.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        print('Load epoch {} succeed！'.format(start_epoch))
        if os.path.exists(os.path.join(dist_path, 'results/losses.npy')):
            losses = np.load(os.path.join(dist_path, 'results/losses.npy'))
            if num_epoch > losses.size:
                losses = np.concatenate((losses, np.zeros(num_epoch - losses.size)))
            print('Load train loss succed！')
        else:
            losses = np.zeros(num_epoch)
        if os.path.exists(os.path.join(dist_path, 'results/test_losses.npy')):
            test_losses = np.load(os.path.join(dist_path, 'results/test_losses.npy'))
            if num_epoch > test_losses.size:
                test_losses = np.concatenate((test_losses, np.zeros(num_epoch - test_losses.size)))
            print('Load test loss succed！')
        else:
            test_losses = np.zeros(num_epoch)
    else:
        start_epoch = 0
        losses = np.zeros(num_epoch)
        test_losses = np.zeros(num_epoch)
        print('Can not load saved model!Training from beginning!')

    para_sk, hopping_index, hopping_info, d, is_hopping, onsite_key, cell_atom_num, onsite_num, orb1_index, orb2_index, orb_num, rvectors, rvectors_all, tensor_E, tensor_eikr, orb_key = utils.batch_index(train_dataloader, traininfos, batch_size)

    for epoch in range(start_epoch + 1, num_epoch + 1):
        for graphs, labels in train_dataloader:
            i = int(labels[0] / batch_size)

            hsk, feat, feato = model(graphs, para_sk[i], is_hopping[i], hopping_index[i], orb_key[i], d[i], onsite_key[i], cell_atom_num[i], onsite_num[i].sum(), orb1_index[i], orb2_index[i])

            b1 = int(hsk.shape[0] / len(labels))
            b2 = int(hopping_info[i].shape[0] / len(labels))
            b3 = int(orb_num[i].shape[0] / len(labels))
            b4 = int(cell_atom_num[i] / len(labels))

            loss = 0
            for j in range(len(labels)):
                HR = utils.construct_hr(hsk[j * b1:(j + 1) * b1], hopping_info[i][j * b2:(j + 1) * b2], orb_num[i][j * b3:(j + 1) * b3], b4, rvectors[i][j])
                reproduced_bands = utils.compute_bands(HR, tensor_eikr[i][j])
                loss += criterion(reproduced_bands[:, 4:12], tensor_E[i][j][:, 4:12])
                
            if is_L1:
                L1 = 0
                for name,param in model.named_parameters():
                    if 'bias' not in name:
                        L1 += torch.norm(param, p=1) * L1_radio
                loss += L1

            if is_L2:
                L2 = 0
                for name,param in model.named_parameters():
                    if 'bias' not in name:
                        L2 += torch.norm(param, p=2) * L2_radio
                loss += L2
            
            opt.zero_grad()
            loss.backward()
            opt.step()

        if is_sch:
            sch.step(loss)

        #test part
        with torch.no_grad():
            test_loss = 0
            for graphs, labels in test_dataloader:
                i = int(labels[0])

                hsk, feat, feato = model(graphs, traininfos[i]['para_sk'], traininfos[i]['is_hopping'], traininfos[i]['hopping_index'], traininfos[i]['orb_key'], traininfos[i]['d'], traininfos[i]['onsite_key'], traininfos[i]['cell_atom_num'], traininfos[i]['onsite_num'].sum(), traininfos[i]['orb1_index'], traininfos[i]['orb2_index'])

                HR = utils.construct_hr(hsk, traininfos[i]['hopping_info'], traininfos[i]['orb_num'], traininfos[i]['cell_atom_num'], traininfos[i]['rvectors'])

                reproduced_bands = utils.compute_bands(HR, traininfos[i]['tensor_eikr'])

                test_loss += criterion(reproduced_bands[:, 4:12], traininfos[i]['tensor_E'][:, 4:12]).item()

        loss_per_epoch[i] = loss.item()

        losses[epoch - 1] = loss_per_epoch.sum() / train_num
        test_losses[epoch - 1] = test_loss / test_num 
        current_lr = opt.param_groups[0]['lr']

        print("Epoch {:05d} | Train_Loss {:.6f} | Test_Loss {:.6f} | Learning_rate {:.6f}" . format(epoch, losses[epoch - 1], test_loss, current_lr))

        if epoch % save_frequncy == 0:

            check_point = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sch.state_dict(),
                    'loss': loss
                    }
            torch.save(check_point, os.path.join(dist_path, 'results/test{}.pkl'.format(epoch)))
            
            torch.save(check_point, latest_point_path)

            np.save(os.path.join(dist_path,'results/losses.npy'), losses)
            np.save(os.path.join(dist_path,'results/test_losses.npy'), test_losses)

    print('trainging OK!')

if __name__ == '__main__':
    train('./')