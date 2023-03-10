## DGP-PGTN:
#### End-to-End Interpretable Disease-Gene Association Prediction with Parallel Graph Transformer Networks

### Quick start

- Run DGP-PGTN:
```
$ python dg_main_sparse.py --epoch 25 --node_dim 32 --lr 0.01 --weight_decay 0.1
```
The essential python packages were listed in ```requirements.txt```.

### Code and data

- `dg_model_sparse.py`: PGTN model
- `gcn.py`: GCN model
- `dg_main_sparse.py`: use the dataset to run DGP-PGTN

#### `data/` directory.  

- `edges_g.pkl`: gene heterogeneous network, which contains five edge types.
- `gene_feature.npy`: node feature of gene heterogeneous networks, which contains 32-dimensional features of 68061 nodes in the gene heterogeneous network
- `edges_d.pkl`: disease heterogeneous network, which contains three edge types.
- `disease_feature.npy`: node feature of disease heterogeneous networks,  which contains 32-dimensional features of 19182 nodes in the gene heterogeneous network
- `result.npy`: disease gene dataset(10000 disease-gene assoications) contains three columns: gene_id, disease_id and their true label.

The file format of the input gene and disease heterogeneous networks is “.pickle”. The data type in the file is a list and the sparse matrices of all association relationships are stored in the list.
For examlpe in the gene heterogeneous networks ```edges_g.pkl```:
```
five types of edges in the gene heterogeneous:	
gene-gene	<with 1051038 stored elements in Compressed Sparse Row format> [gene-gene binary(0 or 1) matrix]
gene-GO		<with 290214 stored elements in Compressed Sparse Row format>  [gene-GO binary(0 or 1) matrix]
GO-gene		<with 290214 stored elements in Compressed Sparse Row format>  [GO-gene binary(0 or 1) matrix]
gene-HPO	<with 182144 stored elements in Compressed Sparse Row format>  [gene-HPO binary(0 or 1) matrix]
HPO-gene	<with 182144 stored elements in Compressed Sparse Row format>  [HPO-gene binary(0 or 1) matrix]
```

### Data statistics 
Disease and gene heterogeneous networks are constructed from five data resources, including HumanNet, Human Phenotype Ontology (HPO), Gene Ontology (GO), Disease Ontology (DO) and DisGeNet
| Network | Type |   Name | Number |
| :-------------: | :----------: |:------------: |:------------: |
| gene network |  node | gene | 21,354 |
|  |       |     GO    | 18,330  |
| |     |   HPO      | 8,153|
| |  relation  | gene-gene     | 1,051,038|
| |    | gene-GO     | 290,214|
| |    | gene-HPO     | 182,144|
|disease network |  node  | disease     |6,453|
|  |       |     gene    | 21,354  |
| |  relation  | disease-disease     | 13,444|
| |  relation  | disease-gene     | 86,297|

### Runtime environment
- Linux terminal and Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz, 8 Core(s), 32 Logical Processor(s)


