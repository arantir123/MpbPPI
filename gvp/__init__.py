import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


# tuple_sum(x, self.dropout[0](dh)), x and dh both include s and V (i.e, (s, V), (s, V))
def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''

    # args = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i)), edge_attr: (batch.edge_s, batch.edge_v)
    dim %= len(args[0][0].shape) # print(dim), dim = 1 (-1%2 = 1)
    s_args, v_args = list(zip(*args))
    # print(len(args[0]), len(args[1]), len(args[2])) 2 2 2
    # print(len(s_args), len(v_args)) 3 3, s_args: node j scalar + edge i_j scalar + node i scalar (similar to v_args)

    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim) # concatenate scalar and vector features separately (on dim=1)

# tuple_index(edge_attr, mask)
def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

# vn = _norm_no_nan(vh, axis=-2)
# vh = sample * 3D coordinates * vector feature num
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps) # the dim to be reduced is vector feature num dim
    return torch.sqrt(out) if sqrt else out

# _split(message, self.vo), self.vo: vector feature output dimension
def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    # GVP(node_in_dim, node_h_dim, activations=(None, None))，decoder output: self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: # whether input vector feature dim exists
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo: # whether output vector feature dim exists
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations # activations=(F.relu, torch.sigmoid)
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x # print(s.size(), v.size()) torch.Size([1573, 6]) torch.Size([1573, 3, 3])
            v = torch.transpose(v, -1, -2) # current dim: sample * 3D coordinates * vector feature num
            vh = self.wh(v) # change the vector feature num to self.h_dim
            vn = _norm_no_nan(vh, axis=-2) # l2 norm, sample * 3D coordinates
            s = self.ws(torch.cat([s, vn], -1)) # merge invariant information of vector features into scalar features, and reduce the dim to self.so (scalar output)
            if self.vo: # if vector feature output is also needed
                v = self.wv(vh) # reduce vector feature dim to self.vo
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: # vector gate only influence the vector feature generation (not directly influence scalar feature generation)
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s)) # use scalar feature to gate vector feature
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True)) # use vector feature to gate vector feature itself
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)

        if self.scalar_act:
            s = self.scalar_act(s) # use relu to further adjust s

        return (s, v) if self.vo else s

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

# representing information propagation part of one GVP-GNN (formulas 3-4, convolution part of GVP-GNN layer)
class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    # self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
    #                     aggr="add" if autoregressive else "mean",
    #                     activations=activations, vector_gate=vector_gate)
    # default n_message = 3, default aggr = 'mean'
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):

        super(GVPConv, self).__init__(aggr=aggr) # setting aggregation scheme in MessagePassing
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        # dimensions in GVPConv:
        # print('in_dims:', in_dims) # in_dims: (100, 16)
        # print('out_dims:', out_dims) # out_dims: (100, 16)
        # print('edge_dims:', edge_dims) # edge_dims: (32, 1)

        # partial usage: https://blog.csdn.net/qq_38335768/article/details/122304391
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        # default module_list: None
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    # input dim, scalar: 100*2+32, vector: 16*2+1, no residue connection in GVP_
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         # output dim
                         (self.so, self.vo),
                         activations=(None, None)))
            else: # default n_layer = 3
                module_list.append(
                    # input dim, scalar: 100*2+32, vector: 16*2+1, no residue connection in GVP_
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims) # activations=(F.relu, torch.sigmoid)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims)) # activations=(F.relu, torch.sigmoid)

                module_list.append(GVP_(out_dims, out_dims, activations=(None, None))) # no activation function
        self.message_func = nn.Sequential(*module_list)

    # h_V = layer(h_V, edge_index, h_E)
    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        # self.propagate will call message，aggregate，update one by one
        # flatten vector feature v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1])
        message = self.propagate(edge_index, s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]), edge_attr=edge_attr)

        # detach message into s and V
        return _split(message, self.vo) 

    # in the setting of flow="source_to_target" in MessagePassing (default), i represents central node, j represents neighboring node
    # edge_attr: tuple (s, V) of `torch.Tensor`
    def message(self, s_i, v_i, s_j, v_j, edge_attr):

        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3) # restore sample * feature num * 3D coordinates
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)

        # concatenate scalar and vector features separately (on dim=1)
        # the features come from node i feature, node j feature and edge i_j feature
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        # print(message[0].size(), message[1].size()) # torch.Size([52410, 232]) 100+100+32 (edge s) torch.Size([52410, 33, 3] 16+16+1 (edge v)

        message = self.message_func(message) # message func: g (3-layer GVP) in formula 3 (actually message includes features coming from node i feature, node j feature and edge i_j feature)

        # merge scalar and vector features (the coordinate dimension is flattened)
        return _merge(*message)

# GVPConvLayer includes residue connection and point-wise layer between graph propagation steps
class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    # representing one GVP-GNN layer
    # GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 # autoregressive mode will not be used in our task as it is used for providing a set of different node embeddings for amino acid inference (CPD task)
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        # formulas 3-4
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean", activations=activations, vector_gate=vector_gate)

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate) # determine the hyper-parameters activations and vector_gate at first

        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)]) # 2 layer norm, process s and V separately
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # formula 5, two feedforward layers (default n_feedforward=2)
        # the scalar dim will increase 4 times and vector dim will increase 2 times in feedforward layers
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1] # node_dim = (100, 16)
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        # in our task, autoregressive mode will not be used
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]

            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward), # (s, v)
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)) # (s, v)

            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        # this mode will be used in our task
        else:
            # call the convolution part (formulas 3-4) in which 3 GVPs are included
            dh = self.conv(x, edge_index, edge_attr)

        # node_mask will not be used in out task
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        # formula 4-5
        # x is the scalar and vector features for current GVP-GNN layer, dh is the output of convolution operation
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) # Sums any number of tuples (s, V) elementwise
        # similar to EGNN here，in message 2/3 MLP/GVP are stacked, and 2 MLP/GVP will be used again during or after the information aggregation
        # print(x[0].size(), x[1].size()) torch.Size([1764, 100]) torch.Size([1764, 16, 3])
        dh = self.ff_func(x) # two GVPs for forward-feedback
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) # the end of one graph propagation step

        # node_mask is not used
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x


# the multi-relational GVPConvLayer (for processing KNN graphs and spatial graphs)
class MR_GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    # representing one GVP-GNN layer
    # GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 # autoregressive mode will not be used in our task as it is used for providing a set of different node embeddings for amino acid inference (CPD task)
                 autoregressive=False, activations=(F.relu, torch.sigmoid), vector_gate=False, graph_cat='sum'):

        super(MR_GVPConvLayer, self).__init__()
        self.graph_cat = graph_cat

        # formulas 3-4
        if self.graph_cat == 'cat':
            self.knn_conv = GVPConv(node_dims, (int(node_dims[0]/2), int(node_dims[1]/2)), edge_dims, n_message,
                                aggr="add" if autoregressive else "mean", activations=activations, vector_gate=vector_gate)
            self.spatial_conv = GVPConv(node_dims,  (int(node_dims[0]/2), int(node_dims[1]/2)), edge_dims, n_message,
                                aggr="add" if autoregressive else "mean", activations=activations, vector_gate=vector_gate)
        else:
            self.knn_conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                                aggr="add" if autoregressive else "mean", activations=activations, vector_gate=vector_gate)
            self.spatial_conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                                aggr="add" if autoregressive else "mean", activations=activations, vector_gate=vector_gate)

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)  # determine the hyper-parameters activations and vector_gate at first

        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # formula 5, two feedforward layers (default n_feedforward=2)
        # the scalar dim will increase 4 times and vector dim will increase 2 times in feedforward layers
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]  # node_dim = (100, 16)
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, extra_edge_index, extra_edge_attr, autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        # in our task, autoregressive mode will not be used
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]

            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),  # (s, v)
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward))  # (s, v)

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        # this mode will be used in our task (and thus we need multiple relation aggregation here)
        else:
            # call the convolution part (formulas 3-4) in which 3 GVPs are included
            if self.graph_cat == 'sum':
                knn_output = self.knn_conv(x, edge_index, edge_attr)
                spatial_output = self.spatial_conv(x, extra_edge_index, extra_edge_attr)
                dh = knn_output[0] + spatial_output[0], knn_output[1] + spatial_output[1]
            elif self.graph_cat == 'mean':
                knn_output = self.knn_conv(x, edge_index, edge_attr)
                spatial_output = self.spatial_conv(x, extra_edge_index, extra_edge_attr)
                dh = (knn_output[0] + spatial_output[0])/2, (knn_output[1] + spatial_output[1])/2
            elif self.graph_cat == 'cat':
                knn_output = self.knn_conv(x, edge_index, edge_attr)
                spatial_output = self.spatial_conv(x, extra_edge_index, extra_edge_attr)
                dh = torch.cat([knn_output[0], spatial_output[0]], dim=-1), torch.cat([knn_output[1], spatial_output[1]], dim=-2)

        # node_mask will not be used in out task
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        # formula 4-5
        # print(x[0].size(), x[1].size(), dh[0].size(), dh[1].size())
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        # node_mask is not used
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x