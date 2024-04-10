import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from dgl.nn import CuGraphGATConv, GATConv, GraphConv
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
from dgl import function as fn
from dgl.base import DGLError

device = 'cuda:0'

class Envelope(nn.Module):  # bessel扩展计算式子

    def __init__(self, exponent):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):

        x_p_0 = x.pow(self.p)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2

        return env_val


class BesselBasisLayer(nn.Module): # bessel扩展

    def __init__(self,
                 num_radial,
                 cutoff = 20.,
                 envelope_exponent = 6,):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    def reset_params(self):
         
         self.frequencies.data = torch.arange(1., self.frequencies.numel() + 1.
                     ).mul_(np.pi)

    def forward(self, d):

        d_scaled = d / self.cutoff
        d_cutoff = self.envelope(d_scaled)

        return d_cutoff * torch.sin(self.frequencies * d_scaled)
    

class MLP(nn.Module):

    def __init__(self, dim_list, activation, dropout=True):
        super(MLP, self).__init__()
        
        self.num_layers = len(dim_list) - 1
        self.activation = activation
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(nn.Linear(dim_list[i], dim_list[i+1])).to(device)
        
        if dropout:
            self.layers.append(nn.Dropout(p=0.1)).to(device)
            self.num_layers += 1

    def forward(self, x):

        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))

        return self.layers[-1](x)


class GATv2Conv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        bessel_cutoff = 4.,
        bessel_exponent = 6,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
    ):
        super(GATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias
                )
        self.attn = nn.Parameter(torch.DoubleTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.prelu = nn.PReLU(num_heads)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bessel = BesselBasisLayer(num_heads, bessel_cutoff, bessel_exponent)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        
        self.reset_parameters()

    def reset_parameters(self):
        
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError()

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]

            coeff = self.bessel(graph.edata['distance'].reshape([-1, 1])).unsqueeze(-1)

            graph.srcdata.update(
                {"el": feat_src}
            )  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({"er": feat_dst})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.prelu(
                graph.edata.pop("e") * coeff
            )  # (num_src_edge, num_heads, out_dim)
            
            e = (
                (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
            )  # (num_edge, num_heads, 1)
            # compute softmax
            
            graph.edata["a"] = self.attn_drop(
                edge_softmax(graph, e)
            )  # (num_edge, num_heads)
            # message passing

            graph.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # print(graph.edata["a"])
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
        
class OrbitalNN(nn.Module):  # 扩展为轨道特征
    def __init__(self, orb_dim_list, activation = nn.LeakyReLU()):
        super(OrbitalNN, self).__init__()

        self.orb_dim_list = orb_dim_list
        self.activation = activation

        self.mlp_list = [MLP(orb_dim_list, activation) for i in range(10)]
    
    def forward(self, feats):

        temp_feat = torch.tensor([]).to(device)
        for i in range(10):
            temp_feat = torch.cat((temp_feat, self.mlp_list[i](feats)), 0)

        return temp_feat


class OnsiteNN(nn.Module): # 从轨道特征变成在位能

    def __init__(self, onsite_dim_list, activation = nn.LeakyReLU()):
        super(OnsiteNN, self).__init__()

        self.onsite_mlp = [MLP(onsite_dim_list, activation) for i in range(10)]
    def forward(self, ofeat, onsite_key, onsite_num):
        '''
        onsite_key 0: 原子序号：代表第几个原子 0 1 2
                   1: 轨道相互作用轨道序号（暂未使用）
                   2: 轨道序号
                   3: onsite矩阵的位置
        ''' 
        onsite = torch.zeros(onsite_num).to(device)

        for i in range(onsite_key[0].size):
            onsite[onsite_key[2][i]] = self.onsite_mlp[onsite_key[1][i]](ofeat[onsite_key[0]][i])
        
        return onsite.flatten()



class HoppingNN(nn.Module): # 从轨道特征生成Slater Koster参量

    def __init__(self,
                 hopping_dim_list1,
                 hopping_dim_list2,
                 is_orb = [1,1,1,1],
                 activation = nn.LeakyReLU()):
        super(HoppingNN, self).__init__()

        # atom features to orbital features (s and p orbitals)

        self.activation = activation
        self.is_orb = is_orb
        self.hopping_dim_list1 = hopping_dim_list1
        self.hopping_dim_list2 = hopping_dim_list2

        self.smlp = MLP([hopping_dim_list1[0]] + hopping_dim_list1[1:], activation)
        self.Smlp = MLP([hopping_dim_list1[0]] + hopping_dim_list1[1:], activation)
        self.pmlp = MLP([hopping_dim_list1[0] * 3] + hopping_dim_list1[1:], activation)
        self.dmlp = MLP([hopping_dim_list1[0] * 5] + hopping_dim_list1[1:], activation)

        self.mlp_list = [MLP(hopping_dim_list2, activation) for i in range(14)]

    def forward(self, feats, hopping_index, atom_num, orb_key, d, ex_d, orb1_index, orb2_index):

        if self.is_orb[0]:
            sfeat = torch.zeros((atom_num, self.hopping_dim_list1[-1])).to(device)
            sfeat[orb1_index[0]] = self.smlp(feats[orb2_index[0]]) *100

        if self.is_orb[1]:
            pfeat = torch.zeros((atom_num, self.hopping_dim_list1[-1])).to(device)
            for oin1, oin2 in zip(orb1_index[1], orb2_index[1]):
                
                pfeat[oin1] = self.pmlp(feats[oin2].flatten())*100

        if self.is_orb[2]:
            dfeat = torch.zeros((atom_num, self.hopping_dim_list1[-1])).to(device)
            for oin1, oin2 in zip(orb1_index[2], orb2_index[2]):
                dfeat[oin1] = self.dmlp(feats[oin2].flatten()) *100

        if self.is_orb[3]:
            Sfeat = torch.zeros((atom_num, self.hopping_dim_list1[-1])).to(device)
            Sfeat[orb1_index[3]] = self.Smlp(feats[orb2_index[3]]) *100

        # 合并成s S p d 轨道特征
        atom1, atom2 = hopping_index[:,0], hopping_index[:,1]

        zeros = torch.zeros((atom1.shape[0],1)).to(device)

        if orb_key[0]: # Vsss
            vfeat = self.mlp_list[0](torch.cat([sfeat[atom1]/(d**2),sfeat[atom2]/(d**2),ex_d], 1))
        else:
            vfeat = zeros
        if orb_key[1]: # Vsps
            vfeat = torch.cat((vfeat, self.mlp_list[1](torch.cat([sfeat[atom1]/(d**2), pfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[2]: # Vpps
            vfeat = torch.cat((vfeat, self.mlp_list[2](torch.cat([pfeat[atom1]/(d**2), pfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[3]: # Vppp
            vfeat = torch.cat((vfeat, self.mlp_list[3](torch.cat([pfeat[atom1]/(d**2), pfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[4]: # Vsds
            vfeat = torch.cat((vfeat, self.mlp_list[4](torch.cat([sfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[5]: # Vpds
            vfeat = torch.cat((vfeat, self.mlp_list[5](torch.cat([pfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[6]: # Vpdp
            vfeat = torch.cat((vfeat, self.mlp_list[6](torch.cat([pfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)

        if orb_key[7]: # Vdds
            vfeat = torch.cat((vfeat, self.mlp_list[7](torch.cat([dfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[8]: # Vddp
            vfeat = torch.cat((vfeat, self.mlp_list[8](torch.cat([dfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[9]: # Vddd
            vfeat = torch.cat((vfeat, self.mlp_list[9](torch.cat([dfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[10]: # VSSs
            vfeat = torch.cat((vfeat, self.mlp_list[10](torch.cat([Sfeat[atom1]/(d**2), Sfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[11]: # VSss
            vfeat = torch.cat((vfeat, self.mlp_list[11](torch.cat([Sfeat[atom1]/(d**2), sfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[12]: # VSps
            vfeat = torch.cat((vfeat, self.mlp_list[12](torch.cat([Sfeat[atom1]/(d**2), pfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        if orb_key[13]: # VSds
            vfeat = torch.cat((vfeat, self.mlp_list[13](torch.cat([Sfeat[atom1]/(d**2), dfeat[atom2]/(d**2),ex_d], 1))),-1)
        else:
            vfeat = torch.cat((vfeat, zeros),-1)
        
        return vfeat


    
class GraphNN(nn.Module):

    def __init__(self,
                dim_list,
                head_list,
                norm = 'both', 
                weight = True, 
                num_heads = 5,
                bias = True, 
                activation = nn.LeakyReLU(),
                ):  
        super(GraphNN, self).__init__()
        
        self.num_layers = len(dim_list) - 1
        self.layers = nn.ModuleList()
        head_list = [1] + head_list
        residual = True
        # residual = False
        negative_slope = 0.5
        share_weights = False

        for i in range(self.num_layers - 1):
           self.layers.append(GATv2Conv(in_feats = dim_list[i] * head_list[i], out_feats = dim_list[i+1], num_heads = head_list[i+1], residual = residual, negative_slope = negative_slope, bias = bias, activation = activation, share_weights = share_weights)).to(device)
           
        self.layers.append(GATv2Conv(in_feats = dim_list[-2]*head_list[-2], out_feats = dim_list[-1], num_heads = head_list[-1], residual = residual, negative_slope = negative_slope, bias = bias, activation = None, share_weights = share_weights)).to(device)


    def forward(self, g, inputs):
        # print(g, inputs.shape)
        
        for i in range(self.num_layers-1) :
            # print(g, i, inputs.shape)
            inputs = self.layers[i](g, inputs).flatten(1)
            
        return self.layers[-1](g, inputs).mean(1)


class WHOLEMODEL(nn.Module):

    def __init__(self,
                  embedding_dim,
                  graph_dim,
                  gnn_dim_list,
                  gnn_head_list,
                  orb_dim_list,
                  onsite_dim_list,
                  hopping_dim_list1,
                  hopping_dim_list2,
                  expander_bessel_dim,
                  expander_bessel_cutoff,
                  atom_num,
                  is_orb = [1,1,1,1],
                  expander_bessel_exponent = 6,
                  orbital_activation = nn.LeakyReLU(negative_slope=0.38),
                  onsite_activation = nn.LeakyReLU(negative_slope=0.38), 
                  hopping_activation = nn.LeakyReLU(negative_slope=0.38),): 
        
        super(WHOLEMODEL, self).__init__()
        
        self.atomic_init_dim = gnn_dim_list[0]
        self.embedding_dim = embedding_dim
        self.graph_dim = embedding_dim
        self.atom_num = atom_num

        self.atomic_feat = nn.Embedding(120, self.embedding_dim) 

        self.gnn = GraphNN(gnn_dim_list, gnn_head_list)
        self.orbnn = OrbitalNN(orb_dim_list, orbital_activation)
        # self.onn = OnsiteNN(onsite_dim_list, onsite_num, onsite_activation)
        self.onn = OnsiteNN(onsite_dim_list, onsite_activation)
        # self.hnn = HoppingNN(hopping_dim_list1, hopping_dim_list2, hopping_activation)
        self.hnn = HoppingNN(hopping_dim_list1, hopping_dim_list2, is_orb, hopping_activation)
        self.expander = BesselBasisLayer(expander_bessel_dim, expander_bessel_cutoff, expander_bessel_exponent)
        self.cutoff = expander_bessel_cutoff
        
    def forward(self, bg, para_sk, is_hopping, hopping_index, orb_key, d, onsite_key, cell_atom_num, onsite_num, orb1_index, orb2_index):

        featstable = bg.ndata['feature'][:, :self.graph_dim]
        featembedding = self.atomic_feat(bg.ndata["species"]).reshape([-1,self.embedding_dim])
        featall = torch.cat((featstable, featembedding), dim=1)

        feat = self.gnn(bg, featall) 
        # o = self.onn(feat[:cell_atom_num])
        feato = self.orbnn(feat)
       
        o = self.onn(feat[:cell_atom_num], onsite_key, onsite_num)
        # h = self.hnn(feat, hopping_index, d, self.expander(d), orb_key)
        h = self.hnn(feato, hopping_index, self.atom_num, orb_key, d, self.expander(d), orb1_index, orb2_index)

        hsk = torch.sum(h*para_sk, dim=1)
        hsk[torch.where(is_hopping==0)[0]] = o
          
        return hsk, feat, feato
