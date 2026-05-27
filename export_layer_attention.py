import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch as th
from sklearn.model_selection import KFold

from load_data import load, remove_graph, get_Fdataset_path
from model import Model
from utils import m2v, set_seed


METAPATH_CANDIDATES = {
    'Kdataset': [
        ['disease_drug', 'drug_disease'],
        ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease'],
        ['disease_drug', 'drug_protein', 'protein_gene', 'gene_pathway', 'pathway_disease'],
        ['disease_pathway', 'pathway_gene', 'gene_protein', 'protein_drug', 'drug_disease'],
    ],
    'Bdataset': [
        ['disease_drug', 'drug_disease'],
        ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease'],
        ['disease_drug', 'drug_drug', 'drug_disease'],
        ['disease_disease', 'disease_drug', 'drug_disease'],
    ],
    'Cdataset': [
        ['disease_drug', 'drug_disease'],
        ['disease_drug', 'drug_drug', 'drug_disease'],
        ['disease_disease', 'disease_drug', 'drug_disease'],
    ],
    'Fdataset': [
        ['disease_drug', 'drug_disease'],
        ['disease_drug', 'drug_drug', 'drug_disease'],
        ['disease_disease', 'disease_drug', 'drug_disease'],
    ],
}

PLOT_SOURCE_ORDER = [
    'H^0_feature',
    'H^R_heterogcn',
    'H^sub_subnetwork',
    'H^G_graph_attention',
    'H^M_metapath',
]

SOURCE_DISPLAY_LABELS = {
    'H^0_feature': 'H^0',
    'H^R_heterogcn': 'H^R',
    'H^sub_subnetwork': 'H^sub',
    'H^G_graph_attention': 'H^G',
    'H^M_metapath': 'H^M',
}


def filter_valid_metapaths(g, metapaths):
    valid_metapaths = []
    for metapath in metapaths:
        try:
            canonical_path = [g.to_canonical_etype(rel) for rel in metapath]
        except Exception:
            continue
        if any(canonical_path[i][2] != canonical_path[i + 1][0] for i in range(len(canonical_path) - 1)):
            continue
        ntypes = {canonical_path[0][0]}
        for stype, _, dtype in canonical_path:
            ntypes.add(stype)
            ntypes.add(dtype)
        if 'drug' in ntypes and 'disease' in ntypes:
            valid_metapaths.append(metapath)
    if len(valid_metapaths) == 0:
        raise ValueError('No valid drug-disease metapath is available for this dataset.')
    return valid_metapaths


def get_feature_and_metapaths(g, dataset):
    feature = {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}
    return feature, filter_valid_metapaths(g, METAPATH_CANDIDATES[dataset])


def select_metapaths_for_ablation(metapaths, metapath_mode):
    if metapath_mode == 'single':
        return metapaths[0]
    return metapaths


def load_drug_disease_matrix(dataset):
    if dataset in ['Kdataset', 'Bdataset']:
        return pd.read_csv('./dataset/{}/{}_baseline.csv'.format(dataset, dataset), header=None).values
    if dataset == 'Cdataset':
        return sio.loadmat('./dataset/Cdataset/Cdataset.mat')['didr'].T
    return pd.read_csv(get_Fdataset_path('drug_disease.csv'), header=None).values


def get_representation_sources(ablation):
    sources = []
    if ablation != 'wo_hm':
        sources.append('H^M_metapath')
    sources.append('H^0_feature')
    if ablation != 'wo_hr':
        sources.append('H^R_heterogcn')
    if ablation != 'wo_hsub':
        sources.append('H^sub_subnetwork')
    if ablation != 'wo_hg':
        sources.append('H^G_graph_attention')
    return sources


def resolve_saved_path(saved_path, seed):
    if os.path.isdir(saved_path):
        return saved_path
    suffixed = '{}_{}'.format(saved_path, seed)
    if os.path.isdir(suffixed):
        return suffixed
    return saved_path


def list_checkpoints(saved_path, nfold):
    checkpoints = glob.glob(os.path.join(saved_path, '*.pth'))
    checkpoints = sorted(checkpoints, key=os.path.getmtime)
    if len(checkpoints) < nfold:
        raise FileNotFoundError(
            'Need at least {} checkpoints in {}, but found {}.'.format(nfold, saved_path, len(checkpoints))
        )
    if len(checkpoints) > nfold:
        print('Found {} checkpoints; using the most recent {} by modified time.'.format(len(checkpoints), nfold))
        checkpoints = checkpoints[-nfold:]
    return checkpoints


def build_cv_splits(df, nfold, seed):
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]

    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    return list(zip(kf.split(data_pos), kf.split(data_neg))), data_pos, data_neg


def semantic_beta(attention_layer, z):
    w = attention_layer.project(z).mean(0)
    return th.softmax(w, dim=0).squeeze(-1).detach().cpu().numpy()


def capture_layer_attention(model, g, feature, drug_emb, disease_emb):
    captured = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, _output):
            captured[name] = semantic_beta(module, inputs[0])
        return hook

    handles.append(model.layer_attention_drug.register_forward_hook(make_hook('drug')))
    handles.append(model.layer_attention_dis.register_forward_hook(make_hook('disease')))

    try:
        with th.no_grad():
            model(g, feature, drug_emb, disease_emb)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def plot_attention(records, output_path):
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [('drug', 'drugs'), ('disease', 'diseases')]
    for ax, (node_type, title) in zip(axes, panels):
        panel = df[df['node_type'] == node_type]
        sources = [source for source in PLOT_SOURCE_ORDER if source in set(panel['source'])]
        labels = [SOURCE_DISPLAY_LABELS[source] for source in sources]
        values = [panel[panel['source'] == source]['attention'].values for source in sources]
        ax.boxplot(values, labels=labels, patch_artist=False)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('attention value')
        ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export final layer-attention weights for drug/disease representations.'
    )
    parser.add_argument('-da', '--dataset', choices=['Kdataset', 'Bdataset', 'Cdataset', 'Fdataset'],
                        default='Kdataset')
    parser.add_argument('-sp', '--saved-path', default='resultK_42',
                        help='Directory containing fold checkpoints. If missing, SCRIPT also tries <saved-path>_<seed>.')
    parser.add_argument('-fo', '--nfold', default=10, type=int)
    parser.add_argument('-se', '--seed', default=42, type=int)
    parser.add_argument('-hf', '--hidden-feats', default=128, type=int)
    parser.add_argument('-he', '--num-heads', default=5, type=int)
    parser.add_argument('-dp', '--dropout', default=0.5, type=float)
    parser.add_argument('--metapath-mode', choices=['attention', 'single', 'first'], default='attention')
    parser.add_argument('--subnet-fusion', choices=['attention', 'first'], default='attention')
    parser.add_argument('--relation-fusion', choices=['attention', 'sum'], default='sum')
    parser.add_argument('--ablation', choices=['none', 'wo_hr', 'wo_hsub', 'wo_hg', 'wo_hm'], default='none')
    parser.add_argument('--device-id', default='0',
                        help='CUDA device id. Use an empty string or "cpu" for CPU.')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory. Defaults to --saved-path.')
    return parser.parse_args()


def main():
    args = parse_args()
    saved_path = resolve_saved_path(args.saved_path, args.seed)
    out_dir = args.out_dir or saved_path
    os.makedirs(out_dir, exist_ok=True)

    if args.device_id.lower() == 'cpu' or args.device_id == '':
        device = th.device('cpu')
    elif th.cuda.is_available():
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('CUDA is not available; falling back to CPU.')
        device = th.device('cpu')

    set_seed(args.seed)
    checkpoints = list_checkpoints(saved_path, args.nfold)
    df = load_drug_disease_matrix(args.dataset)
    splits, data_pos, data_neg = build_cv_splits(df, args.nfold, args.seed)
    sources = get_representation_sources(args.ablation)
    records = []

    for fold, (((_train_pos_idx, test_pos_idx), (_train_neg_idx, _test_neg_idx)), checkpoint) in enumerate(
            zip(splits, checkpoints), start=1):
        print('Extracting fold {} from {}'.format(fold, os.path.basename(checkpoint)))
        test_pos_id = data_pos[test_pos_idx]

        g = load(args.dataset)
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)
        feature, metapaths = get_feature_and_metapaths(g, args.dataset)
        metapaths_for_m2v = select_metapaths_for_ablation(metapaths, args.metapath_mode)

        if args.ablation == 'wo_hm':
            drug_emb, disease_emb = None, None
        else:
            drug_emb, disease_emb = m2v(g, metapaths_for_m2v, device=device, emb_dim=args.hidden_feats)

        model = Model(
            etypes=g.etypes,
            ntypes=g.ntypes,
            in_feats=feature['drug'].shape[1],
            hidden_feats=args.hidden_feats,
            num_heads=args.num_heads,
            dropout=args.dropout,
            metapath_mode=args.metapath_mode,
            subnet_fusion=args.subnet_fusion,
            relation_fusion=args.relation_fusion,
            ablation=args.ablation,
        ).to(device)
        model.load_state_dict(th.load(checkpoint, map_location=device))
        model.eval()

        captured = capture_layer_attention(model, g, feature, drug_emb, disease_emb)
        for node_type in ['drug', 'disease']:
            values = captured[node_type]
            if len(values) != len(sources):
                raise RuntimeError(
                    '{} attention length {} does not match representation list {}.'.format(
                        node_type, len(values), len(sources)
                    )
                )
            for h_idx, (source, attention) in enumerate(zip(sources, values)):
                records.append({
                    'fold': fold,
                    'node_type': node_type,
                    'H': 'H{}'.format(h_idx),
                    'source': source,
                    'attention': float(attention),
                    'checkpoint': os.path.basename(checkpoint),
                })

    values_path = os.path.join(out_dir, 'layer_attention_values.csv')
    summary_path = os.path.join(out_dir, 'layer_attention_summary.csv')
    plot_path = os.path.join(out_dir, 'layer_attention_boxplot.png')

    values = pd.DataFrame(records)
    source_order = {source: index for index, source in enumerate(PLOT_SOURCE_ORDER)}
    values['plot_order'] = values['source'].map(source_order)
    values = values.sort_values(['node_type', 'plot_order', 'fold'])
    values.to_csv(values_path, index=False)
    values.groupby(['node_type', 'plot_order', 'H', 'source'])['attention'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index().sort_values(['node_type', 'plot_order']).to_csv(summary_path, index=False)
    plot_attention(values.to_dict('records'), plot_path)

    print('Saved raw values to {}'.format(values_path))
    print('Saved summary to {}'.format(summary_path))
    print('Saved figure to {}'.format(plot_path))


if __name__ == '__main__':
    main()
