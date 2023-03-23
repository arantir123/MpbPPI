import torch
import torch.nn as nn
from . import GVP, GVPConvLayer, LayerNorm, tuple_index, _merge, MR_GVPConvLayer
from torch.distributions import Categorical
from torch_scatter import scatter_mean


class EquiPPIModel(torch.nn.Module):
    '''
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch`
    
    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers used in the gvp
    :param drop_rate: rate to use in all dropout layers
    '''

    # gvp.models.EquiPPIModel((29, 9), node_dim, (32, 1), edge_dim).to(device), node_dim = (100, 16), edge_dim = (32, 1) (intermediate embedding dimension)
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1):
    
        super(EquiPPIModel, self).__init__()

        self.W_v = nn.Sequential(
            # (29, 9) -> (100, 16)
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            # (32, 1) -> (32, 1)
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # in one GVP-GNN layer (GVPConvLayer, by default there are three GVPs for information aggregation and two GVPs for feedforward processing)
        # the main hyper-parameters to be tuned here are layer num and output dim (node_h_dim, edge_h_dim, num_layers)
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))

    # logits = gvp(h_V, batch.edge_index, h_E, (X_ca_label, sidec_label, SASA_label))
    def forward(self, h_V, edge_index, h_E):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param labels: labels for multi-task pretraining
        '''

        # use one layer GVP to do dimension transformation for node embeddings and edge embeddings
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # use multiple-layer GVP-GNN (the layer number is specified by num_layers hyperparameter)
        # in the default setting, in one GVP-GNN, the neighboring information to be aggregated (neighboring node feature and corresponding edge feature) will be processed by 3-layer GVPs
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E) # similar to EGNN, the edge embedding will not update along with GNN here

        # prepare to implement decoder for multi-task pretraining
        # for a task in which the predictor is used to (can be used to) predict the both dog and cat category (for training, based on their corresponding pictures), the predictor number could be 1
        # otherwise, like our task that needs to predict muliple labels (with some connections) simultaneously for the same sample/input, maybe it is better to set independent predictors for each task

        # if vo=0 in GVP, GVP only output scalar feature part (after interacting with vector feature), example: self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))
        # if vo!=0, GVP also output vector feature part, and vector_gate only influences the generation of vector features (replace self-interaction with the interaction with scalar features)
        # here we consider MLP to predict objectives at first, as what GVP generates are scalar + vector, which still needs to be further processed
        
        encoder_embeddings = _merge(h_V[0], h_V[1]) # s: torch.Size([2825, 100]) and V: torch.Size([2825, 16, 3]) are obtained from the encoder_layers
        
        return encoder_embeddings


# the variant of EquiPPIModel that gets the last two GNN layers as the model output (for simulating the GeoPPI mode)
class EquiPPIModel_(torch.nn.Module):
    '''
    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch`

    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers used in the gvp
    :param drop_rate: rate to use in all dropout layers
    '''

    # gvp.models.EquiPPIModel((29, 9), node_dim, (32, 1), edge_dim).to(device), node_dim = (100, 16), edge_dim = (32, 1) (intermediate embedding dimension)
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1):
        super(EquiPPIModel_, self).__init__()

        self.W_v = nn.Sequential(
            # (29, 9) -> (100, 16)
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            # (32, 1) -> (32, 1)
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # in one GVP-GNN layer (GVPConvLayer, by default there are three GVPs for information aggregation and two GVPs for feedforward processing)
        # the main hyper-parameters to be tuned here are layer num and output dim (node_h_dim, edge_h_dim, num_layers)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        self.encoder_layer_num = len(self.encoder_layers)

    # logits = gvp(h_V, batch.edge_index, h_E, (X_ca_label, sidec_label, SASA_label))
    def forward(self, h_V, edge_index, h_E):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param labels: labels for multi-task pretraining
        '''

        # use one layer GVP to do dimension transformation for node embeddings and edge embeddings
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # use multiple-layer GVP-GNN (the layer number is specified by num_layers hyperparameter)
        # in the default setting, in one GVP-GNN, the neighboring information to be aggregated (neighboring node feature and corresponding edge feature) will be processed by 3-layer GVPs
        counter = 0
        for layer in self.encoder_layers:
            counter += 1
            h_V = layer(h_V, edge_index, h_E)  # similar to EGNN, the edge embedding will not update along with GNN here
            if counter == self.encoder_layer_num - 1:
                h_V_ = h_V

        # prepare to implement decoder for multi-task pretraining
        # for a task in which the predictor is used to (can be used to) predict the both dog and cat category (for training, based on their corresponding pictures), the predictor number could be 1
        # otherwise, like our task that needs to predict muliple labels (with some connections) simultaneously for the same sample/input, maybe it is better to set independent predictors for each task

        # if vo=0 in GVP, GVP only output scalar feature part (after interacting with vector feature), example: self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))
        # if vo!=0, GVP also output vector feature part, and vector_gate only influences the generation of vector features (replace self-interaction with the interaction with scalar features)
        # here we consider MLP to predict objectives at first, as what GVP generates are scalar + vector, which still needs to be further processed

        encoder_embeddings = _merge(h_V[0], h_V[1])  # s: torch.Size([2825, 100]) and V: torch.Size([2825, 16, 3]) are obtained from the encoder_layers
        encoder_embeddings_ = _merge(h_V_[0], h_V_[1])

        return encoder_embeddings, encoder_embeddings_
    
    
class multi_task_predictor(nn.Module):
    # input_dropout: the dropout for the first layer of MLP in each predictor, med_dropout: the dropout for the intermediate layers of MLP in each predictor
    def __init__(self, whether_SASA, in_feats, layer_list, input_dropout=0.0, med_dropout=0.0, whether_AA_prediction=False, whether_sidec_prediction=False):
        super(multi_task_predictor, self).__init__()
        self.whether_SASA = whether_SASA
        self.whether_AA_prediction = whether_AA_prediction
        self.whether_sidec_prediction = whether_sidec_prediction

        backb_predictor = nn.ModuleList()
        self.backb_predictor = predictor_creator(backb_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_sidec_prediction:
            sidec_predictor = nn.ModuleList()
            self.sidec_predictor = predictor_creator(sidec_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_SASA == True: # treating SASA prediction as an auxiliary task
            SASA_predictor = nn.ModuleList()
            layer_list[-1] = 1 # all dim in predictor: in_feats + layer_list
            self.SASA_predictor = predictor_creator(SASA_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_AA_prediction == True:
            self.AA_classifier = torch.nn.Linear(in_feats, 20)
            if input_dropout > 0:
                self.AA_dropout = nn.Dropout(input_dropout)
            else:
                self.AA_dropout = lambda x: x

    def forward(self, encoder_embeddings, new_ca_label):
        # 1. CA coordinate denoising task
        predicted_ca_coord = encoder_embeddings
        # print(encoder_embeddings)
        for layer in self.backb_predictor:
            predicted_ca_coord = layer(predicted_ca_coord)
        predicted_ca_coord = new_ca_label + predicted_ca_coord # torch.Size([368, 3])

        # 2. side chain coordinate information denoising task
        if self.whether_sidec_prediction:
            # comparsion of torch.repeat and torch.expand: https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969
            # when only expanding only one dimension, tensor.expand should be faster than tensor.repeat as tensor.expand will not create a new memory for it
            # but tensor.expand cannot expand multiple dimensions simultaneouly, in other words:
            # expand() will never allocate new memory. And so require the expanded dimension to be of size 1 (can be more efficient in this case).
            # repeat() will always allocate new memory and the repeated dimension can be of any size.
            predicted_sidec_info = encoder_embeddings.unsqueeze(-2).expand(-1, 5, -1) # SASA_label: torch.Size([338, 5, 3])
            # print(encoder_embeddings)
            for layer in self.sidec_predictor:
                predicted_sidec_info = layer(predicted_sidec_info)

        # 3. SASA prediction task
        if self.whether_SASA == True:
            predicted_SASA = encoder_embeddings
            for layer in self.SASA_predictor:
                predicted_SASA = layer(predicted_SASA)

        # 4. AA prediction task
        if self.whether_AA_prediction == True:
            predicted_AA = encoder_embeddings
            predicted_AA = self.AA_classifier(self.AA_dropout(predicted_AA))

        if self.whether_SASA == True and self.whether_AA_prediction == True and self.whether_sidec_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA, predicted_AA
        elif self.whether_SASA == True and self.whether_AA_prediction == False and self.whether_sidec_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA
        elif self.whether_SASA == False and self.whether_AA_prediction == True and self.whether_sidec_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_AA
        elif self.whether_SASA == True and self.whether_AA_prediction == True and self.whether_sidec_prediction == False:
            return predicted_ca_coord, predicted_SASA, predicted_AA
        elif self.whether_SASA == False and self.whether_AA_prediction == False and self.whether_sidec_prediction == True:
            return predicted_ca_coord, predicted_sidec_info
        elif self.whether_SASA == True and self.whether_AA_prediction == False and self.whether_sidec_prediction == False:
            return predicted_ca_coord, predicted_SASA
        elif self.whether_SASA == False and self.whether_AA_prediction == True and self.whether_sidec_prediction == False:
            return predicted_ca_coord, predicted_AA
        else:
            return predicted_ca_coord


class multi_task_predictor_cg(nn.Module):
    # input_dropout: the dropout for the first layer of MLP in each predictor, med_dropout: the dropout for the intermediate layers of MLP in each predictor
    def __init__(self, whether_SASA, in_feats, layer_list, input_dropout=0.0, med_dropout=0.0, whether_AA_prediction=False, whether_CG_prediction=False, CG_encoding_type='sincos'):
        super(multi_task_predictor_cg, self).__init__()
        self.whether_SASA = whether_SASA
        self.whether_AA_prediction = whether_AA_prediction
        self.whether_CG_prediction = whether_CG_prediction
        self.CG_encoding_type = CG_encoding_type
        if self.whether_CG_prediction:
            if self.CG_encoding_type == 'sincos':
                output_dim = 5 + 5
            elif self.CG_encoding_type == 'onehot':
                output_dim = 32
            else:
                output_dim = 5

        backb_predictor = nn.ModuleList()
        self.backb_predictor = predictor_creator(backb_predictor, in_feats, layer_list, input_dropout, med_dropout)

        sidec_predictor = nn.ModuleList()
        self.sidec_predictor = predictor_creator(sidec_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_SASA == True: # treating SASA prediction as an auxiliary regression task
            SASA_predictor = nn.ModuleList()
            layer_list[-1] = 1 # all dim in predictor: in_feats + layer_list
            self.SASA_predictor = predictor_creator(SASA_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_AA_prediction == True:
            self.AA_classifier = torch.nn.Linear(in_feats, 20)
            if input_dropout > 0:
                self.AA_dropout = nn.Dropout(input_dropout)
            else:
                self.AA_dropout = lambda x: x

        if self.whether_CG_prediction == True: # treating CG prediction as an auxiliary regression task
            if CG_encoding_type == 'onehot':
                self.CG_predictor = torch.nn.Linear(in_feats, output_dim)
                if input_dropout > 0:
                    self.CG_dropout = nn.Dropout(input_dropout)
                else:
                    self.CG_dropout = lambda x: x
            else:
                CG_predictor = nn.ModuleList()
                layer_list[-1] = output_dim
                self.CG_predictor = predictor_creator(CG_predictor, in_feats, layer_list, input_dropout, med_dropout)

    def forward(self, encoder_embeddings, new_ca_label):
        # 1. CA coordinate denoising task
        predicted_ca_coord = encoder_embeddings
        # print(encoder_embeddings)
        for layer in self.backb_predictor:
            predicted_ca_coord = layer(predicted_ca_coord)
        predicted_ca_coord = new_ca_label + predicted_ca_coord # torch.Size([368, 3])

        # 2. side chain coordinate information denoising task
        # comparsion of torch.repeat and torch.expand: https://discuss.pytorch.org/t/torch-repeat-and-torch-expand-which-to-use/27969
        # when only expanding only one dimension, tensor.expand should be faster than tensor.repeat as tensor.expand will not create a new memory for it
        # but tensor.expand cannot expand multiple dimensions simultaneouly, in other words:
        # expand() will never allocate new memory. And so require the expanded dimension to be of size 1 (can be more efficient in this case).
        # repeat() will always allocate new memory and the repeated dimension can be of any size.
        predicted_sidec_info = encoder_embeddings.unsqueeze(-2).expand(-1, 5, -1) # SASA_label: torch.Size([338, 5, 3])
        # print(encoder_embeddings)
        for layer in self.sidec_predictor:
            predicted_sidec_info = layer(predicted_sidec_info)

        # 3. SASA prediction task
        if self.whether_SASA == True:
            predicted_SASA = encoder_embeddings
            for layer in self.SASA_predictor:
                predicted_SASA = layer(predicted_SASA)

        # 4. AA prediction task
        if self.whether_AA_prediction == True:
            predicted_AA = encoder_embeddings
            predicted_AA = self.AA_classifier(self.AA_dropout(predicted_AA))

        # 5. CG prediction task
        if self.whether_CG_prediction == True:
            predicted_CG = encoder_embeddings
            if self.CG_encoding_type == 'onehot':
                predicted_CG = self.CG_predictor(self.CG_dropout(predicted_CG))
            else:
                for layer in self.CG_predictor:
                    predicted_CG = layer(predicted_CG)

        # self.whether_CG_prediction == True
        if self.whether_SASA == True and self.whether_AA_prediction == True and self.whether_CG_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA, predicted_AA, predicted_CG
        elif self.whether_SASA == True and self.whether_AA_prediction == False and self.whether_CG_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA, predicted_CG
        elif self.whether_SASA == False and self.whether_AA_prediction == True and self.whether_CG_prediction == True:
            return predicted_ca_coord, predicted_sidec_info, predicted_AA, predicted_CG
        # # self.whether_CG_prediction == False
        elif self.whether_SASA == True and self.whether_AA_prediction == True and self.whether_CG_prediction == False:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA, predicted_AA
        elif self.whether_SASA == True and self.whether_AA_prediction == False and self.whether_CG_prediction == False:
            return predicted_ca_coord, predicted_sidec_info, predicted_SASA
        elif self.whether_SASA == False and self.whether_AA_prediction == True and self.whether_CG_prediction == False:
            return predicted_ca_coord, predicted_sidec_info, predicted_AA
        else: # all false, only retain CA and sidec prediction tasks
            return predicted_ca_coord, predicted_sidec_info


class affinity_change_decoder(nn.Module):
    # input_dropout: the dropout for the first layer of MLP in the decoder, med_dropout: the dropout for the intermediate layers of MLP in the decoder
    def __init__(self, in_feats, layer_list, input_dropout=0.0, med_dropout=0.0):
        super(affinity_change_decoder, self).__init__()

        affinity_decoder = nn.ModuleList()
        self.affinity_decoder = predictor_creator(affinity_decoder, in_feats, layer_list, input_dropout, med_dropout)

    def forward(self, screened_embeddings):
        for layer in self.affinity_decoder:
            screened_embeddings = layer(screened_embeddings)

        return screened_embeddings


def predictor_creator(module_list, in_feats, layer_list, input_dropout, med_dropout):
    for i in range(len(layer_list)):
        if i == 0:
            module_list.append(torch.nn.Linear(in_feats, layer_list[i]))
            module_list.append(torch.nn.ReLU())
            module_list.append(nn.Dropout(input_dropout))
        elif i == len(layer_list) - 1:  # the last layer
            module_list.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
        else:  # the intermediate layers
            module_list.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            module_list.append(torch.nn.ReLU())
            module_list.append(nn.Dropout(med_dropout))
    return module_list


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# the variant of EquiPPIModel that uses both KNN graph and radius graph for message passing
class MR_EquiPPIModel(torch.nn.Module):
    '''
    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch`

    :param node_in_dim: node dimensions in input graph
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers used in the gvp
    :param drop_rate: rate to use in all dropout layers
    '''

    # gvp.models.EquiPPIModel((29, 9), node_dim, (32, 1), edge_dim).to(device), node_dim = (100, 16), edge_dim = (32, 1) (intermediate embedding dimension)
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1, graph_cat='sum'):
        super(MR_EquiPPIModel, self).__init__()

        self.W_v = nn.Sequential(
            # (29, 9) -> (100, 16)
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
            LayerNorm(node_h_dim)
        )
        self.W_e = nn.Sequential(
            # (32, 1) -> (32, 1)
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
            LayerNorm(edge_h_dim)
        )

        # in one GVP-GNN layer (GVPConvLayer, by default there are three GVPs for information aggregation and two GVPs for feedforward processing)
        # the main hyper-parameters to be tuned here are layer num and output dim (node_h_dim, edge_h_dim, num_layers)
        self.encoder_layers = nn.ModuleList(
            MR_GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, graph_cat=graph_cat)
            for _ in range(num_layers))

    # logits = gvp(h_V, batch.edge_index, h_E, (X_ca_label, sidec_label, SASA_label))
    def forward(self, h_V, edge_index, h_E, extra_edge_index, extra_h_E):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param labels: labels for multi-task pretraining
        '''

        # use one layer GVP to do dimension transformation for node embeddings and edge embeddings
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        # use multiple-layer GVP-GNN (the layer number is specified by num_layers hyperparameter)
        # in the default setting, in one GVP-GNN, the neighboring information to be aggregated (neighboring node feature and corresponding edge feature) will be processed by 3-layer GVPs
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E, extra_edge_index, extra_h_E)  # similar to EGNN, the edge embedding will not update along with GNN here

        # prepare to implement decoder for multi-task pretraining
        # for a task in which the predictor is used to (can be used to) predict the both dog and cat category (for training, based on their corresponding pictures), the predictor number could be 1
        # otherwise, like our task that needs to predict muliple labels (with some connections) simultaneously for the same sample/input, maybe it is better to set independent predictors for each task

        # if vo=0 in GVP, GVP only output scalar feature part (after interacting with vector feature), example: self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))
        # if vo!=0, GVP also output vector feature part, and vector_gate only influences the generation of vector features (replace self-interaction with the interaction with scalar features)
        # here we consider MLP to predict objectives at first, as what GVP generates are scalar + vector, which still needs to be further processed

        encoder_embeddings = _merge(h_V[0], h_V[1])  # s: torch.Size([2825, 100]) and V: torch.Size([2825, 16, 3]) are obtained from the encoder_layers

        return encoder_embeddings


class multi_task_predictor_wosidec(nn.Module):
    # input_dropout: the dropout for the first layer of MLP in each predictor, med_dropout: the dropout for the intermediate layers of MLP in each predictor
    def __init__(self, whether_SASA, in_feats, layer_list, input_dropout=0.0, med_dropout=0.0, whether_AA_prediction=False, remove_SASA_all=False):
        super(multi_task_predictor_wosidec, self).__init__()
        self.whether_SASA = whether_SASA
        self.whether_AA_prediction = whether_AA_prediction
        self.remove_SASA_all = remove_SASA_all

        backb_predictor = nn.ModuleList()
        self.backb_predictor = predictor_creator(backb_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_SASA == True and self.remove_SASA_all == False: # treating SASA prediction as an auxiliary task
            SASA_predictor = nn.ModuleList()
            layer_list[-1] = 1 # all dim in predictor: in_feats + layer_list
            self.SASA_predictor = predictor_creator(SASA_predictor, in_feats, layer_list, input_dropout, med_dropout)

        if self.whether_AA_prediction == True:
            self.AA_classifier = torch.nn.Linear(in_feats, 20)
            if input_dropout > 0:
                self.AA_dropout = nn.Dropout(input_dropout)
            else:
                self.AA_dropout = lambda x: x

    def forward(self, encoder_embeddings, new_ca_label):
        # 1. CA coordinate denoising task
        predicted_ca_coord = encoder_embeddings
        # print(encoder_embeddings)
        for layer in self.backb_predictor:
            predicted_ca_coord = layer(predicted_ca_coord)
        predicted_ca_coord = new_ca_label + predicted_ca_coord # torch.Size([368, 3])

        # 2. SASA prediction task
        if self.whether_SASA == True and self.remove_SASA_all == False:
            predicted_SASA = encoder_embeddings
            for layer in self.SASA_predictor:
                predicted_SASA = layer(predicted_SASA)

        # 3. AA prediction task
        if self.whether_AA_prediction == True:
            predicted_AA = encoder_embeddings
            predicted_AA = self.AA_classifier(self.AA_dropout(predicted_AA))

        if self.whether_SASA == True and self.remove_SASA_all == False and self.whether_AA_prediction == True:
            return predicted_ca_coord, predicted_SASA, predicted_AA
        elif self.whether_SASA == True and self.remove_SASA_all == False and self.whether_AA_prediction == False:
            return predicted_ca_coord, predicted_SASA
        elif (self.whether_SASA == False or self.remove_SASA_all == True) and self.whether_AA_prediction == True:
            return predicted_ca_coord, predicted_AA
        else:
            return predicted_ca_coord






