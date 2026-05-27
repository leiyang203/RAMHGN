import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default='0', type=str,
                    help='Set the device (GPU ids).')
parser.add_argument('-da', '--dataset', type=str, choices=['Kdataset', 'Bdataset', 'Cdataset', 'Fdataset'], default='Kdataset',
                    help='Set the data set for training.')
parser.add_argument('-sp', '--saved_path', type=str,
                    help='Path to save training results', default='resultK')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed')
# Training Arguments
parser.add_argument('-fo', '--nfold', default=10, type=int,
                    help='The number of k in K-folds Validation')
parser.add_argument('-ep', '--epoch', default=4000, type=int,
                    help='Number of epochs for training')
parser.add_argument('-lr', '--learning_rate', default=0.003, type=float,
                    help='learning rate to use')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay to use')
parser.add_argument('-pa', '--patience', default=300, type=int,
                    help='Early Stopping argument')
parser.add_argument('--pos-weight-scale', '--pos_weight_scale', dest='pos_weight_scale',
                    default=0.25, type=float,
                    help='Scale factor for BCE positive class weight. The final pos_weight is '
                         'negative samples / positive samples * pos_weight_scale.')
# Model Arguments
parser.add_argument('-hf', '--hidden_feats', default=128, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-he', '--num_heads', default=5, type=int,
                    help='Number of attention heads the model has')
parser.add_argument('-dp', '--dropout', default=0.5, type=float,
                    help='The rate of dropout layer')
parser.add_argument('--metapath-mode', '--metapath_mode', dest='metapath_mode',
                    choices=['attention', 'single', 'first'], default='attention',
                    help='Metapath ablation mode: attention uses all metapaths with attention; '
                         'single only computes the first metapath; first computes all metapaths '
                         'but keeps only the first embedding.')
parser.add_argument('--subnet-fusion', '--subnet_fusion', dest='subnet_fusion',
                    choices=['attention', 'first'], default='attention',
                    help='Subnetwork ablation mode: attention fuses all subnet branches; '
                         'first keeps only the first branch for each node type.')
parser.add_argument('--relation-fusion', '--relation_fusion', dest='relation_fusion',
                    choices=['attention', 'sum'], default='sum',
                    help='Relation aggregation mode in HeteroGCN layers. attention learns relation '
                         'weights; sum uses the original MRDDA-style summed relation aggregation.')
parser.add_argument('--ablation', type=str,
                    choices=['none', 'wo_hr', 'wo_hsub', 'wo_hg', 'wo_hm'], default='none',
                    help='Representation-level ablation switch. none keeps the full model; '
                         'wo_hr excludes H^R; wo_hsub excludes H^sub; '
                         'wo_hg excludes H^G; wo_hm excludes H^M.')

args = parser.parse_args()
args.saved_path = args.saved_path + '_' + str(args.seed)
