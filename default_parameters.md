# Default Adjustable Parameters

| Category | Short option | Long option | Default | Type | Choices | Description |
| --- | --- | --- | --- | --- | --- | --- |
| General | `-id` | `--device_id` | `0` | `str` | - | GPU device id. Empty value means CPU mode in the current code. |
| General | `-da` | `--dataset` | `Kdataset` | `str` | `Kdataset`, `Bdataset`, `Cdataset` | Dataset used for training and evaluation. |
| General | `-sp` | `--saved_path` | `resultK` | `str` | - | Base path for saving training results. The code appends `_<seed>`, so the effective default is `resultK_42`. |
| General | `-se` | `--seed` | `42` | `int` | - | Global random seed. |
| Training | `-fo` | `--nfold` | `10` | `int` | - | Number of folds for K-fold cross-validation. |
| Training | `-ep` | `--epoch` | `4000` | `int` | - | Maximum number of training epochs. |
| Training | `-lr` | `--learning_rate` | `0.003` | `float` | - | Learning rate. |
| Training | `-wd` | `--weight_decay` | `0.0001` | `float` | - | Weight decay for Adam optimizer. |
| Training | `-pa` | `--patience` | `300` | `int` | - | Early stopping patience. |
| Training | - | `--pos-weight-scale`, `--pos_weight_scale` | `0.25` | `float` | - | Scale factor for the positive class weight in BCE loss. Final `pos_weight = negative_samples / positive_samples * pos_weight_scale`. |
| Model | `-hf` | `--hidden_feats` | `128` | `int` | - | Hidden feature dimension. |
| Model | `-he` | `--num_heads` | `5` | `int` | - | Number of attention heads. |
| Model | `-dp` | `--dropout` | `0.5` | `float` | - | Dropout rate. |
| Model | - | `--metapath-mode`, `--metapath_mode` | `attention` | `str` | `attention`, `single`, `first` | Metapath ablation mode. |
| Model | - | `--subnet-fusion`, `--subnet_fusion` | `attention` | `str` | `attention`, `first` | Subnetwork fusion ablation mode. |
| Model | - | `--relation-fusion`, `--relation_fusion` | `sum` | `str` | `attention`, `sum` | Relation aggregation mode in HeteroGCN layers. |
| Model | - | `--ablation` | `none` | `str` | `none`, `wo_hr`, `wo_hsub`, `wo_hg`, `wo_hm` | Representation-level ablation switch for excluding `H^R`, `H^sub`, `H^G`, or `H^M` from layer attention fusion. |
