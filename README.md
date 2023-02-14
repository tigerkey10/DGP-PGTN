## DGP-PGTN:
#### End-to-End Interpretable Disease-Gene Association Prediction with Parallel Graph Transformer Networks

### Run

To predict disease-gene association:
```
$ python dg_main_sparse.py --epoch 25 --node_dim 32 --lr 0.01 --weight_decay 0.1
```
The essential python packages were listed in ```requirements.txt```.

### Code and data

#### 
- `dg_model_sparse.py`: PGTN model
- `gcn.py`: GCN model
- `dg_main_sparse.py`: use the dataset to run DGP-PGTN


#### Data sample `/data` directory.  
The file format of the input gene and disease heterogeneous networks is “.pickle”, the data type in the file is a list and the sparse matrices of all association relationships are stored in the list.
- `edges_g.pkl`: gene heterogeneous network
- `gene_feature.npy`: node feature of gene heterogeneous networks
- `edges_d.pkl`: disease heterogeneous network
- `disease_feature.npy`: node feature of disease heterogeneous networks
- `result.npy`: disease gene dataset contains three columns: gene_id, disease_id and their assoications

### Train DGP-PGTN on a Linux terminal and Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz, 8 Core(s), 32 Logical Processor(s).

### Prediction server for the broader reach of the research paper. 
- http://nefunlp.cn/
