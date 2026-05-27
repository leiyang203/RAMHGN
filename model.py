import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl
from dgl.nn.pytorch import GATConv


class InnerProductDecoder(nn.Module):
    """Decoder layer for prediction"""

    def __init__(self, input_dim=None, dropout=0.4):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.linear1 = nn.Linear(128, input_dim)
        self.linear2 = nn.Linear(128, input_dim)

    def forward(self, feature):
        R = self.dropout(feature['drug'])
        D = self.dropout(feature['disease'])
        D = self.weights(D)
        outputs = R @ D.T
        return outputs


class RelationAttention(nn.Module):
    """Relation-level attention for aggregating different edge semantics."""

    def __init__(self, in_feats, hidden_size=128):
        super(RelationAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        beta = torch.softmax(self.project(z), dim=1)
        return (beta * z).sum(1)


class Node_Embedding(nn.Module):
    """Relation-aware HeteroGCN layer."""

    def __init__(self, in_feats, out_feats, dropout, rel_names, relation_fusion='attention'):
        super().__init__()
        self.relation_fusion = relation_fusion
        self.convs = nn.ModuleDict()
        for rel in rel_names:
            graphconv = dglnn.GraphConv(in_feats, out_feats, allow_zero_in_degree=True)
            nn.init.xavier_normal_(graphconv.weight)
            self.convs[rel] = graphconv
        self.dropout = nn.Dropout(p=dropout)
        self.relation_attention = RelationAttention(out_feats)
        self.bn_layer = nn.BatchNorm1d(out_feats)
        self.prelu = nn.PReLU()

    def forward(self, graph, inputs, bn=False, dp=False):
        relation_h = {ntype: [] for ntype in inputs}
        for stype, etype, dtype in graph.canonical_etypes:
            if etype not in self.convs or stype not in inputs or dtype not in inputs:
                continue
            rel_graph = graph[(stype, etype, dtype)]
            if stype == dtype:
                rel_h = self.convs[etype](rel_graph, inputs[stype])
            else:
                rel_h = self.convs[etype](rel_graph, (inputs[stype], inputs[dtype]))
            relation_h.setdefault(dtype, []).append(rel_h)

        h = {}
        for ntype, rel_outputs in relation_h.items():
            if not rel_outputs:
                continue
            if self.relation_fusion == 'sum':
                h[ntype] = torch.stack(rel_outputs, dim=0).sum(0)
            elif len(rel_outputs) == 1:
                h[ntype] = rel_outputs[0]
            elif self.relation_fusion == 'attention':
                h[ntype] = self.relation_attention(torch.stack(rel_outputs, dim=1))
            else:
                raise ValueError('Unsupported relation fusion mode: {}'.format(self.relation_fusion))

        if bn and dp:
            h = {k: self.prelu(self.dropout(self.bn_layer(v))) for k, v in h.items()}
        elif dp:
            h = {k: self.prelu(self.dropout(v)) for k, v in h.items()}
        elif bn:
            h = {k: self.prelu(self.bn_layer(v)) for k, v in h.items()}
        else:
            h = {k: self.prelu(v) for k, v in h.items()}
        return h


class SemanticAttention(nn.Module):
    """The base attention mechanism used in layer attention."""

    def __init__(self, in_feats, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class SubnetworkEncoder(nn.Module):

    def __init__(self, etypes, ntypes, in_feats, out_feats, dropout,
                 subnet_fusion='attention', relation_fusion='attention'):
        super(SubnetworkEncoder, self).__init__()
        self.ntypes = ntypes
        self.etypes = set(etypes)
        self.subnet_fusion = subnet_fusion
        self.relation_fusion = relation_fusion
        self.subnet_specs = []
        self.subnet_layers = nn.ModuleDict()
        self.subnet_attention = nn.ModuleDict({
            ntype: SemanticAttention(in_feats=out_feats) for ntype in ntypes
        })
        self._add_subnet('drug_disease', ['drug', 'disease'],
                         ['drug_drug', 'drug_disease', 'disease_drug', 'disease_disease'],
                         in_feats, out_feats, dropout)
        self._add_subnet('drug_protein', ['drug', 'protein'],
                         ['drug_drug', 'drug_protein', 'protein_drug', 'protein_protein'],
                         in_feats, out_feats, dropout)
        self._add_subnet('protein_gene', ['protein', 'gene'],
                         ['protein_protein', 'protein_gene', 'gene_protein', 'gene_gene'],
                         in_feats, out_feats, dropout)
        self._add_subnet('gene_pathway', ['gene', 'pathway'],
                         ['gene_gene', 'gene_pathway', 'pathway_gene', 'pathway_pathway'],
                         in_feats, out_feats, dropout)
        self._add_subnet('pathway_disease', ['pathway', 'disease'],
                         ['pathway_pathway', 'pathway_disease', 'disease_pathway', 'disease_disease'],
                         in_feats, out_feats, dropout)

    def _add_subnet(self, name, node_types, rel_names, in_feats, out_feats, dropout):
        if all(ntype in self.ntypes for ntype in node_types) and all(rel in self.etypes for rel in rel_names):
            self.subnet_layers[name] = Node_Embedding(in_feats, out_feats, dropout, rel_names,
                                                      relation_fusion=self.relation_fusion)
            self.subnet_specs.append((name, node_types, rel_names))

    def forward(self, g, h, bn=False, dp=False):
        new_h = {ntype: [] for ntype in self.ntypes}

        for name, node_types, rel_names in self.subnet_specs:
            g_ = g.edge_type_subgraph(rel_names)
            subnet_inputs = {ntype: h[ntype] for ntype in node_types}
            h_ = self.subnet_layers[name](g_, subnet_inputs, bn, dp)
            for ntype in node_types:
                if ntype in h_:
                    new_h[ntype].append(h_[ntype])

        for ntype in self.ntypes:
            if len(new_h[ntype]) == 0:
                continue
            if self.subnet_fusion == 'first' or len(new_h[ntype]) == 1:
                h[ntype] = new_h[ntype][0]
            elif self.subnet_fusion == 'attention':
                h[ntype] = self.subnet_attention[ntype](torch.stack(new_h[ntype], dim=1))
            else:
                raise ValueError('Unsupported subnet fusion mode: {}'.format(self.subnet_fusion))

        return h


class Graph_attention(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, dropout):
        super().__init__()
        self.gat = GATConv(in_feats, out_feats, num_heads,
                                 dropout, dropout,
                                 activation=nn.PReLU(),
                                 allow_zero_in_degree=True)
        self.gat.reset_parameters()
        self.linear = nn.Linear(in_feats * num_heads, out_feats)
        self.prelu = nn.PReLU()
        self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, graph, inputs, bn=False):
        num_dis = graph.num_nodes('disease')
        num_drug = graph.num_nodes('drug')
        new_g = dgl.to_homogeneous(graph)
        new_h = torch.cat([i for i in inputs.values()], dim=0)
        new_h = self.gat(new_g, new_h)
        new_h = self.prelu(torch.mean(new_h, dim=1))
        if bn:
            return self.bn_layer(new_h[:num_dis]), self.bn_layer(new_h[num_dis:num_drug + num_dis])
        return new_h[:num_dis], new_h[num_dis:num_drug + num_dis]


class Model(nn.Module):
    """The overall MRDDA architecture."""

    def __init__(self, etypes, ntypes, in_feats, hidden_feats, num_heads, dropout,
                 metapath_mode='attention', subnet_fusion='attention',
                 relation_fusion='attention', ablation='none'):
        super(Model, self).__init__()
        valid_ablations = {'none', 'wo_hr', 'wo_hsub', 'wo_hg', 'wo_hm'}
        if ablation not in valid_ablations:
            raise ValueError('Unsupported ablation mode: {}'.format(ablation))
        self.ntypes = ntypes
        self.metapath_mode = metapath_mode
        self.ablation = ablation
        if 'drug' in ntypes:
            self.drug_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.drug_linear.weight)
        if 'disease' in ntypes:
            self.disease_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.disease_linear.weight)
        if 'protein' in ntypes:
            self.protein_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.protein_linear.weight)
        if 'gene' in ntypes:
            self.gene_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.gene_linear.weight)
        if 'pathway' in ntypes:
            self.pathway_linear = nn.Linear(in_feats, hidden_feats)
            nn.init.xavier_normal_(self.pathway_linear.weight)

        self.HeteroGCN_layer1 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes,
                                               relation_fusion=relation_fusion)
        self.HeteroGCN_layer2 = Node_Embedding(hidden_feats, hidden_feats, dropout, etypes,
                                               relation_fusion=relation_fusion)
        self.subnet_layer = SubnetworkEncoder(etypes, ntypes, hidden_feats, hidden_feats, dropout,
                                              subnet_fusion=subnet_fusion,
                                              relation_fusion=relation_fusion)
        self.gat_layer = Graph_attention(hidden_feats, hidden_feats, num_heads, dropout)
        self.metapath_attention_drug = SemanticAttention(hidden_feats)
        self.metapath_attention_dis = SemanticAttention(hidden_feats)
        self.layer_attention_drug = SemanticAttention(hidden_feats)
        self.layer_attention_dis = SemanticAttention(hidden_feats)
        self.predict = InnerProductDecoder(hidden_feats)

    def _fuse_metapath_embedding(self, emb, attention_layer):
        if emb.dim() != 3:
            return emb
        if self.metapath_mode in ['single', 'first']:
            return emb[:, 0, :]
        if self.metapath_mode == 'attention':
            return attention_layer(emb)
        raise ValueError('Unsupported metapath mode: {}'.format(self.metapath_mode))

    def forward(self, g, x, mdrug, mdis):
        drug_emb_list, dis_emb_list = [], []
        if self.ablation != 'wo_hm':
            if mdrug is None or mdis is None:
                raise ValueError('Metapath embeddings are required unless ablation is wo_hm.')
            mdrug = self._fuse_metapath_embedding(mdrug, self.metapath_attention_drug)
            mdis = self._fuse_metapath_embedding(mdis, self.metapath_attention_dis)
            drug_emb_list.append(mdrug)
            dis_emb_list.append(mdis)

        h = {}
        for ntype in self.ntypes:
            h[ntype] = x[ntype]
        h['drug'] = self.drug_linear(h['drug'])
        h['disease'] = self.disease_linear(h['disease'])
        if 'protein' in self.ntypes:
            h['protein'] = self.protein_linear(h['protein'])
        if 'gene' in self.ntypes:
            h['gene'] = self.gene_linear(h['gene'])
        if 'pathway' in self.ntypes:
            h['pathway'] = self.pathway_linear(h['pathway'])
        drug_emb_list.append(h['drug'])
        dis_emb_list.append(h['disease'])

        if self.ablation != 'wo_hr':
            h = self.HeteroGCN_layer1(g, h, bn=True, dp=True)
            h = self.HeteroGCN_layer2(g, h, bn=True, dp=True)
            drug_emb_list.append(h['drug'])
            dis_emb_list.append(h['disease'])

        if self.ablation != 'wo_hsub':
            h = self.subnet_layer(g, h, bn=False, dp=True)
            drug_emb_list.append(h['drug'])
            dis_emb_list.append(h['disease'])

        if self.ablation != 'wo_hg':
            h['disease'], h['drug'] = self.gat_layer(g, h)
            drug_emb_list.append(h['drug'])
            dis_emb_list.append(h['disease'])

        h['drug'] = self.layer_attention_drug(torch.stack(drug_emb_list, dim=1))
        h['disease'] = self.layer_attention_dis(torch.stack(dis_emb_list, dim=1))

        return self.predict(h)
