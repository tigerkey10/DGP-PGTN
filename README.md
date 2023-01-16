# DGP-PGTN
End-to-End Interpretable Disease-Gene Association Prediction with Parallel Graph Transformer Networks
All process
-Run ./dg_main_sparse.py You can run the entire model
Code and data
dg_model.py PGTN model
gcn.py: GCN model
main.py: use the dataset to run DGP-PGTN
data sample ./data directory
edges_g.pkl: gene heterogeneous network
gene_feature.npy: node feature of gene heterogeneous networks
edges_d.pkl: disease heterogeneous network
disease_feature.npy: node feature of disease heterogeneous networks
result.npy: disease-gene association
