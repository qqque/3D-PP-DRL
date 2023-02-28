from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class SpatialPositionEncoding(nn.Module):
    def __init__(self, d_model, bin_size_x,bin_size_y):
        super().__init__()
        not_mask = torch.ones((1, bin_size_x, bin_size_y))
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        pos_feature = d_model // 2
        dim_t = torch.arange(pos_feature, dtype=torch.float32)
        dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc') / pos_feature))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        self.register_buffer('pos', pos)

    def forward(self, x):
        # x: batch_size x seq_len x d_model 
        return x + self.pos


def binary_encoder_old(x,binary_dim=8):
    if binary_dim == 1:
        return x
    x_size = torch.tensor(x.size())
    x_size[-1] = -1
    x_encoding = torch.tensor(np.unpackbits(
        np.array(x.cpu(), np.uint8)), dtype=torch.float32).reshape(*x_size).cuda()
    return x_encoding

def binary_encoder(x,binary_dim):
    # x: size x 1
    if binary_dim == 1:
        x_encoding = x
    else:
        binary_list = []
        divide = x
        for i in range(binary_dim):
            binary = divide % 2
            binary_list.insert(0,binary)
            divide = torch.div(divide,2,rounding_mode='trunc')

        x_encoding = torch.stack(binary_list,-1).flatten(-2,-1)
    return x_encoding

def batch_index_equal(tensor_batch, index_batch, value):
    tensor_batch = tensor_batch * 1
    index = index_batch + torch.arange(0, tensor_batch.size(0)).to(index_batch.device) * tensor_batch.size(1)
    tensor_batch.flatten(0, 1)[index.to(torch.long)] = value

    return tensor_batch


def batch_index_select(tensor_batch, index_batch):
    index = index_batch + torch.arange(0, tensor_batch.size(0)).to(index_batch.device) * tensor_batch.size(1)
    tensor_select = tensor_batch.flatten(0, 1).index_select(0, index.to(torch.int))

    return tensor_select


class TransformerEnc(nn.Module):
    def __init__(self, d_model=128, n_head=4, d_inner=1024, n_layers=2, dropout=0):
        super().__init__()
        layers = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, batch_first=True)
        self.transformer = TransformerEncoder(layers, n_layers)

    def forward(self, x, mask=None):
        out = self.transformer(x, src_key_padding_mask=mask)

        return out


class TransformerDec(nn.Module):
    def __init__(self, d_model=128, n_head=8, d_inner=1024, n_layers=2, dropout=0):
        super().__init__()
        layers = TransformerDecoderLayer(d_model, n_head, d_inner, dropout, batch_first=True)
        self.transformer = TransformerDecoder(layers, n_layers)

    def forward(self, query, key, query_mask=None, key_mask=None):
        out = self.transformer(query, key, tgt_key_padding_mask=query_mask, memory_key_padding_mask=key_mask)

        return out


class BoxShapeEmbedd(nn.Module):
    def __init__(self, d_model=128, d_inner=256,binary_dim=8):
        super().__init__()
        self.binary_dim= binary_dim
        self.encoder = nn.Sequential(nn.Linear(binary_dim, d_model), nn.Tanh(),
                                     nn.Linear(d_model, d_inner), nn.Tanh(),
                                     nn.Linear(d_inner, d_model))

    def forward(self, box_state):
        # bin_state: batch_size x box_num x 3
        bs, bn, _ = box_state.size()
        box_state = binary_encoder(box_state,self.binary_dim).view(bs, bn, 3, self.binary_dim)
        box_embedd = self.encoder(box_state)
        box_embedd = torch.mean(box_embedd, -2, keepdim=False)  # bs x bn x d_model

        return box_embedd


class BoxEncoder(nn.Module):
    def __init__(self, d_model, shape_embedd=True,binary_dim=8):
        super().__init__()
        if shape_embedd:
            self.embedd = BoxShapeEmbedd(d_model=d_model, d_inner=d_model * 2,binary_dim=binary_dim)
        else:
            self.embedd = nn.Linear(3*binary_dim, d_model)
        self.transformer = TransformerEnc(d_model)

    def forward(self, box_state, box_mask=None):
        box_embedd = self.embedd(box_state)
        box_encoder = self.transformer(box_embedd, box_mask)

        return box_encoder


class BinEncoder(nn.Module):
    def __init__(self, d_model, bin_size_x,bin_size_y, feature_dim,binary_dim):
        super().__init__()
        self.binary_dim = binary_dim
        self.pos_encoder_bin = SpatialPositionEncoding(d_model, bin_size_x,bin_size_y)
        self.embedd = nn.Linear(feature_dim, d_model)
        self.transformer = TransformerEnc(d_model)

    def forward(self, bin_state):
        bin_embedd = binary_encoder(bin_state,self.binary_dim)
        bin_embedd = self.embedd(bin_embedd)
        bin_embedd = self.pos_encoder_bin(bin_embedd)
        bin_encoder = self.transformer(bin_embedd)

        return bin_encoder


class BoxSelectAction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.transformer = TransformerDec(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, box_encoder, bin_encoder):
        decoder = self.transformer(box_encoder, bin_encoder)
        a_logits = self.fc(decoder).squeeze(-1)

        return a_logits


class BoxSelectActionPF(nn.Module):
    def __init__(self, d_model=128, d_inner=256, feature_dim=56,binary_dim=8):
        super().__init__()
        self.binary_dim = binary_dim
        self.position_embedd = nn.Sequential(nn.Linear(feature_dim, d_model), nn.Tanh(),
                                             nn.Linear(d_model, d_inner), nn.Tanh(),
                                             nn.Linear(d_inner, d_model))
        self.transformer = TransformerDec(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, box_encoder, position_feature, position_encoder):
        position_feature = binary_encoder(position_feature,self.binary_dim)
        position_embedd = self.position_embedd(position_feature)
        position_embedd = position_embedd + position_encoder
        decoder = self.transformer(box_encoder, position_embedd)
        a_logits = self.fc(decoder).squeeze(-1)

        return a_logits


class PositionActionPF(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.position_decoder = TransformerDec(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, bin_encoder, box_encoder):
        position_decoder = self.position_decoder(bin_encoder, box_encoder)
        a_logits = self.fc(position_decoder).squeeze(-1)

        return a_logits


class PositionAction(nn.Module):
    def __init__(self, d_model,binary_dim):
        super().__init__()
        self.box_select_embedd = BoxShapeEmbedd(d_model, d_model * 2,binary_dim)
        self.box_other_encoder = BoxEncoder(d_model,binary_dim=binary_dim)
        self.box_encoder = TransformerDec(d_model)
        self.position_decoder = TransformerDec(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, box_select_shape, box_other_state, bin_encoder, box_other_mask):
        # box_select_shape: batch_size x 1 3
        box_select_shape = box_select_shape
        box_select_embedd = self.box_select_embedd(box_select_shape)
        box_other_encoder = self.box_other_encoder(box_other_state, box_other_mask)
        box_encoder = self.box_encoder(box_select_embedd, box_other_encoder)
        position_decoder = self.position_decoder(bin_encoder, box_encoder)
        a_logits = self.fc(position_decoder).squeeze(-1)

        return a_logits


class RotationAction(nn.Module):
    def __init__(self, d_model=128, d_inner=256, feature_dim=56,binary_dim=8):
        super().__init__()
        self.binary_dim=binary_dim
        self.box_rot_embedd = nn.Linear(3*binary_dim, d_model)
        self.position_embedd = nn.Sequential(nn.Linear(feature_dim, d_model), nn.Tanh(),
                                             nn.Linear(d_model, d_inner), nn.Tanh(),
                                             nn.Linear(d_inner, d_model))
        self.transformer = TransformerDec(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, box_rot_state, position_feature, position_encoder):
        position_feature = binary_encoder(position_feature,self.binary_dim)
        box_rot_feature = binary_encoder(box_rot_state,self.binary_dim)
        position_embedd = self.position_embedd(position_feature)
        position_embedd = position_embedd + position_encoder
        box_rot_embedd = self.box_rot_embedd(box_rot_feature)
        decoder = self.transformer(box_rot_embedd, position_embedd)
        a_logits = self.fc(decoder).squeeze(-1)

        return a_logits


class ActorNetwork(nn.Module):
    def __init__(self,
                 bin_size_x,bin_size_y,
                 box_num,
                 plane_feature_num = 6,
                 d_model=128,
                 nhead=4,
                 d_hid=128,
                 nlayers=2,
                 binary_dim=9):
        super().__init__()
        self.box_num = box_num
        self.binary_dim=binary_dim
        self.bin_state_dim = (1+plane_feature_num) * binary_dim
        self.box_encoder = BoxEncoder(d_model,binary_dim=binary_dim)
        self.bin_encoder = BinEncoder(d_model, bin_size_x,bin_size_y, self.bin_state_dim,self.binary_dim)
        self.softmax_box_index = nn.Softmax(-1)
        self.softmax_position_index = nn.Softmax(-1)
        self.rotation_action = RotationAction(d_model, d_model * 2, self.bin_state_dim,self.binary_dim)
        self.softmax_rotation = nn.Softmax(-1)

        self.position_action = PositionActionPF(d_model)
        self.select_box_action = BoxSelectActionPF(d_model, feature_dim=self.bin_state_dim,binary_dim=self.binary_dim)


    def forward(self, state, action_old=None):

        bin_state = state[0]
        box_state = state[1]
        # max_index = state[2]
        packing_mask = state[2].flatten(3, 4)  # batch_size x residual_box_num x
        device = bin_state.device
        batch_size = bin_state.size()[0]
        # get box and bin encoder
        box_mask = box_state[:, :, 0] < 0
        box_encoder = self.box_encoder(box_state, box_mask)
        bin_state_flat = bin_state.flatten(1, 2)
        bin_encoder = self.bin_encoder(bin_state_flat)

        # get position x and y
        position_logits = self.position_action(bin_encoder, box_encoder)
        rotation_matrix = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                                        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                                        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]],
                                       dtype=torch.float32).to(box_state.device)

        box_rotation_shape_all = torch.matmul(box_state, rotation_matrix).view(batch_size, -1, 6, 3)

        position_mask = packing_mask.all(1).all(1)
        box_select_mask_all = packing_mask.all(2).transpose(1, 2)
        box_rotation_mask_all = packing_mask.permute(0, 3, 1, 2).to(dtype=torch.bool)

        pos_mask_softmax = torch.where(position_mask, -1e9, 0.0)
        position_index_prob = self.softmax_position_index(position_logits + pos_mask_softmax)

        if action_old != None:
            position_index = torch.as_tensor(action_old[1], dtype=torch.int64).to(device)
        else:
            position_index = torch.multinomial(
                position_index_prob, num_samples=1, replacement=True).squeeze(-1)

        position_feature = batch_index_select(bin_state_flat, position_index).unsqueeze(-2)  # batch_size x 1 x 8

        position_encoder = batch_index_select(bin_encoder, position_index).unsqueeze(-2)

        box_select_mask = batch_index_select(box_select_mask_all, position_index)
        box_index_logits = self.select_box_action(box_encoder, position_feature, position_encoder)
        box_softmax_mask = torch.where(box_select_mask, -1e9, 0.0)
        box_index_prob = self.softmax_box_index(box_index_logits + box_softmax_mask)

        if action_old != None:
            box_index = torch.as_tensor(action_old[0], dtype=torch.int64).to(device)
        else:
            box_index = torch.multinomial(
                box_index_prob, num_samples=1, replacement=True).squeeze(-1)

        box_rotation_shape = batch_index_select(box_rotation_shape_all, box_index)
        rotation_logits = self.rotation_action(box_rotation_shape, position_feature, position_encoder)

        rotation_mask_all = batch_index_select(box_rotation_mask_all, position_index)
        rotation_mask = batch_index_select(rotation_mask_all, box_index)
        rot_mask_softmax = torch.where(rotation_mask, -1e9, 0.0)
        rotation_prob = self.softmax_rotation(rotation_logits + rot_mask_softmax)

        rotation_index = torch.multinomial(
            rotation_prob, num_samples=1, replacement=True).squeeze(-1)

        prob = (box_index_prob, position_index_prob, rotation_prob)
        action = (box_index, position_index, rotation_index)

        return prob, action

    def get_action(self, state):
        a_prob, action = self.forward(state)

        return action, a_prob

    def get_logprob_entropy(self, state, action):
        irxy_prob, _ = self.forward(state, action_old=action)
        dist_irxy = [torch.distributions.Categorical(prob) for prob in irxy_prob]
        log_prob_irxy = [dist_irxy[i].log_prob(action[i]) for i in range(len(dist_irxy))]
        log_prob = sum(log_prob_irxy)
        entropy_irxy = [dist.entropy().mean() for dist in dist_irxy]
        entropy = sum(entropy_irxy)
        return log_prob, entropy

    def get_old_logprob(self, action, a_prob):
        dist_irxy = [torch.distributions.Categorical(prob) for prob in a_prob]
        log_prob_irxy = [dist_irxy[i].log_prob(action[i]) for i in range(len(dist_irxy))]
        log_prob = sum(log_prob_irxy)
        return log_prob

    def get_position_mask(self, box_rotation_shape, bin_size, max_index):

        batch_size = box_rotation_shape.size(0)

        y_residual_size = torch.zeros((batch_size, bin_size, bin_size)) + torch.arange(bin_size, 0, -1)
        x_residual_size = y_residual_size.transpose(1, 2)
        position_residual_size = torch.stack([x_residual_size, y_residual_size], 3).flatten(1, 2).unsqueeze(-2).to(
            box_rotation_shape.device)  # batch_size x bin_size**2 x 1 x 2
        box_rotation_shape = box_rotation_shape[:, :, :2].unsqueeze(-3)
        if self.large_bin_size:
            position_residual_size = position_residual_size * 10
            max_index_size = torch.cat([torch.floor(max_index / 10), max_index % 10], -1)  # batch_size x 100 x 1 x 2
            position_residual_size = position_residual_size - max_index_size
        can_be_packed = (box_rotation_shape <= position_residual_size).all(-1)

        position_mask = can_be_packed.any(-1)

        return ~position_mask, ~can_be_packed

    def get_position_mask_pf(self, box_rotation_shape, bin_size, max_index, box_mask, limit_height=None,
                             bin_height=None):

        batch_size = box_rotation_shape.size(0)

        y_residual_size = torch.zeros((batch_size, bin_size, bin_size)) + torch.arange(bin_size, 0, -1)
        x_residual_size = y_residual_size.transpose(1, 2)
        position_residual_size = torch.stack([x_residual_size, y_residual_size], 3).flatten(1, 2).unsqueeze(
            -2).unsqueeze(-2).to(box_rotation_shape.device)  # batch_size x bin_size**2 x 1 x 1 x 2
        box_plane_shape = box_rotation_shape[:, :, :, :2].unsqueeze(1)  # batch_size x 1 x box_num x 6 x 2
        box_height_shape = box_rotation_shape[:, :, :, 2].unsqueeze(1)  # batch_size x 1 x box_num x 6 x 1
        if self.large_bin_size:
            position_residual_size = position_residual_size * 10
            max_index_size = torch.cat([torch.floor(max_index / 10), max_index % 10], -1).unsqueeze(
                -2)  # batch_size x 100 x 1 x 1 x 2
            position_residual_size = position_residual_size - max_index_size
        box_rotation_mask = (box_plane_shape <= position_residual_size).all(-1)  # batch_size x 100 x box_num x 6
        if limit_height is not None:
            # bin_height: batch_size x bin_size**2 x 1
            bin_height = bin_height.unsqueeze(-1).unsqueeze(-1)  # batch_size x bin_size**2 x 1 x 1
            box_height_mask = (box_height_shape + bin_height) <= limit_height  # batch_size x bin_size**2 x box_num x 6
            box_rotation_mask = box_rotation_mask * box_height_mask
        box_select_mask = box_rotation_mask.any(-1) * (~box_mask.unsqueeze(1))

        position_mask = box_select_mask.any(-1)

        return ~position_mask, ~box_select_mask, ~box_rotation_mask

    def _get_xy_mask(self, box_shape):
        bs = box_shape.size(0)
        box_length = torch.cat([box_shape[:, 0:1]] * self.bin_size, dim=1)
        box_width = torch.cat([box_shape[:, 1:2]] * self.bin_size, dim=1)
        bin_length = torch.cat(
            [torch.arange(self.bin_size, 0, -1).unsqueeze(0)] * bs, dim=0).cuda()
        bin_width = torch.cat(
            [torch.arange(self.bin_size, 0, -1).unsqueeze(0)] * bs, dim=0).cuda()
        x_bool = (box_length <= bin_length)
        y_bool = (box_width <= bin_width)
        x_mask = torch.where(x_bool, 0., float("-inf"))
        y_mask = torch.where(y_bool, 0., float("-inf"))
        return x_mask, y_mask


class CriticNetwork(nn.Module):
    def __init__(self, bin_size_x,bin_size_y, box_num, plane_feature_num=6, d_model=128,binary_dim=9):
        super().__init__()
        self.box_num = box_num

        self.bin_state_dim = (1 + plane_feature_num) * binary_dim

        self.box_encoder = BoxEncoder(d_model,binary_dim=binary_dim)
        self.bin_encoder = BinEncoder(d_model, bin_size_x,bin_size_y,self.bin_state_dim,binary_dim)
        self.select_box_action = BoxSelectAction(d_model)

        # bin_decoder -> value
        self.fc_v1 = nn.Linear(d_model, 1)
        self.fc_v2 = nn.Linear(bin_size_x * bin_size_y, bin_size_x)
        self.fc_v3 = nn.Linear(bin_size_x, 1)

    def forward(self, state):
        # bin_state: bz×bin_size×bin_size×1
        # box_state: bz×box_num×18

        bin_state = state[0]
        box_state = state[1]
        device = bin_state.device
        bin_size = bin_state.size(1)

        # get box index
        box_mask = box_state[:, :, 0] < 0
        box_encoder = self.box_encoder(box_state, box_mask)
        bin_state_flat = bin_state.flatten(1, 2)
        bin_encoder = self.bin_encoder(bin_state_flat,)
        value = self.select_box_action(bin_encoder, box_encoder)

        value = F.relu(self.fc_v2(value))
        value = self.fc_v3(value)

        return value
