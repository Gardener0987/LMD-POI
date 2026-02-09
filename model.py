import torch
import math
from torch import nn
from torch.nn import Module
from torch_geometric.nn import GatedGraphConv,GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class POIGraph3(Module):
    def __init__(self,
                 n_nodes,
                 hidden_size,
                 gcn_layers: int = 2,
                 ggnn_layers: int = 2):

        super(POIGraph3, self).__init__()
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes

        self.embedding = nn.Embedding(self.n_nodes, self.hidden_size)
        
        self.gcn_layers = nn.ModuleList()
        for i in range(gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
        
        self.ggnn = GatedGraphConv(hidden_size, num_layers=ggnn_layers)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight is None:
                continue
            weight.data.uniform_(-stdv, stdv)

    def score_edges(self, hidden, edge_index):

        src, dst = edge_index[0], edge_index[1]
        h_src = hidden[src]
        h_dst = hidden[dst]
        scores = F.cosine_similarity(h_src, h_dst, dim=1)

        return scores

    def per_node_topk(self, edge_index, scores, num_nodes, k=10, mode='in'):
        src, dst = edge_index
        E = scores.size(0)
        mask = torch.zeros(E, dtype=torch.bool, device=scores.device)
        node_edge_counts = {}
        for v in range(num_nodes):
            if mode == 'in':
                idx = (dst == v).nonzero(as_tuple=True)[0]
            else:
                idx = (src == v).nonzero(as_tuple=True)[0]
            node_edge_counts[v] = idx.numel()
            if idx.numel() == 0:
                continue
                
            if idx.numel() <= k:
                mask[idx] = True
            else:
                local_scores = scores[idx]
                topk_local = torch.topk(local_scores, k=k).indices
                mask[idx[topk_local]] = True

        return edge_index[:, mask]

    def forward(self, inputs, A):
        device = A.device if isinstance(A, torch.Tensor) else torch.device('cpu')
        hidden = self.embedding(inputs.to(device))
        hidden_init =hidden

        for gcn_layer in self.gcn_layers:
            hidden = F.relu(gcn_layer(hidden, A))
        
        hidden_gcn = hidden  
        edge_scores = self.score_edges(hidden_gcn, A)
        edge_index_denoised = self.per_node_topk(A, edge_scores, self.n_nodes, k=15, mode='in')
        hidden_final = self.ggnn(hidden_init, edge_index_denoised)
        return hidden_final

class UserEmbeddings1(nn.Module):
    def __init__(self, num_users, embedding_dim, poi_embed_dim):
        super(UserEmbeddings1, self).__init__()
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.poi_project = nn.Linear(poi_embed_dim, embedding_dim)
        self.gcn1 = GCNConv(embedding_dim, embedding_dim)
        self.gcn2 = GCNConv(embedding_dim, embedding_dim)
        self.act = nn.LeakyReLU(0.2)
        self.fusion_linear = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, user_idx, poi_embeddings, edge_index):
       
        all_user_embeds = self.user_embedding.weight 
        poi_feats = self.poi_project(poi_embeddings) 
        x = torch.cat([all_user_embeds, poi_feats], dim=0)
        x = self.gcn1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.act(x)
        updated_user_embeds = x[:self.num_users]
        batch_user_embeds = updated_user_embeds[user_idx]
        initial_user_embed = self.user_embedding(user_idx)
        out = batch_user_embeds + initial_user_embed
        out = self.fusion_linear(out)
        return out
   
class CatEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CatEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, cat_idx):
        return self.cat_embedding(cat_idx)

class FuseEmbeddings(nn.Module):
    def __init__(self, embed_dim1, embed_dim2):
        super(FuseEmbeddings, self).__init__()
        embed_dim = embed_dim1 + embed_dim2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embed1, embed2):
        x_in = torch.cat((embed1, embed2),0)   
        x = self.fuse_embed(x_in)
        x = self.leaky_relu(x)
        x = x + x_in
        x = self.layer_norm(x)  
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.sin = SineActivation(1, out_dim)

    def forward(self, x):
        x = self.sin(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_times,num_cats, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, num_times)
        self.decoder_cat = nn.Linear(embed_size, num_cats)  
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)


    def forward(self, src, src_mask, src_key_padding_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)        
        return out_poi,out_time,out_cat
    


