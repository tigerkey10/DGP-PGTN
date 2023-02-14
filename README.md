## DGP-PGTN:
End-to-End Interpretable Disease-Gene Association Prediction with Parallel Graph Transformer Networks

### All process

To predict disease-gene association, run:
```
$ python dg_main_sparse.py --epoch 25 --node_dim 32 --lr 0.01 --weight_decay 0.1
```

### Code and data

#### 
- `dg_model_sparse.py`: PGTN model
- `gcn.py`: GCN model
- `dg_main_sparse.py`: use the dataset to run DGP-PGTN


#### data sample `/data` directory.  The file format of the input gene and disease heterogeneous networks is “.pickle”, the data type in the file is a list and the sparse matrices of all association relationships are stored in the list.



- `edges_g.pkl`: gene heterogeneous network
- `gene_feature.npy`: node feature of gene heterogeneous networks
- `edges_d.pkl`: disease heterogeneous network
- `disease_feature.npy`: node feature of disease heterogeneous networks
- `result.npy`: disease-gene association

#### train DGP-PGTN on a Linux terminal on Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz, 8 Core(s), 32 Logical Processor(s).

#### prediction server for the broader reach of the research paper. 
- [Disease-gene association prediction server](http://nefunlp.cn/), which contains gene_id, disease_id, true label and our predict.
