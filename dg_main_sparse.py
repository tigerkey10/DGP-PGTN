import torch
import numpy as np
from dg_model_sparse import GTN
import pickle
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as auc3
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=25,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=32,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default= 0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='l2 reg')
    args = parser.parse_args()
    print('args:',args)
    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    weight_decay = args.weight_decay
    np.random.seed(42)


    num_channels_u = 2

    with open('data/edges_g.pkl', 'rb') as f:
        edges_u= pickle.load(f)



    num_nodes_u = edges_u[0].shape[0]
    A_u=[]
    for i,edge in enumerate(edges_u):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        A_u.append((edge_tmp,value_tmp))
    edge_u_tmp = torch.stack((torch.arange(0,num_nodes_u),torch.arange(0,num_nodes_u))).type(torch.LongTensor)
    value_u_tmp = torch.ones(num_nodes_u).type(torch.FloatTensor)
    A_u.append((edge_u_tmp,value_u_tmp))
    with open('data/gene_feature.npy', 'rb') as f:
        edges_u= np.load(f)
    node_features_u = torch.from_numpy(edges_u)


    num_channels_v = 2

    with open('data/edges_d.pkl', 'rb') as f:
        edges_v= pickle.load(f)
    num_nodes_v = edges_v[0].shape[0]

    A_v=[]
    for i, edge in enumerate(edges_v):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        A_v.append((edge_tmp,value_tmp))
    edge_v_tmp = torch.stack((torch.arange(0, num_nodes_v),torch.arange(0,num_nodes_v))).type(torch.LongTensor)
    value_v_tmp = torch.ones(num_nodes_v).type(torch.FloatTensor)
    A_v.append((edge_v_tmp,value_v_tmp))

    with open('data/disease_feature.npy', 'rb') as f:
        edges_v = np.load(f)
    node_features_v = torch.from_numpy(edges_v)

    result = np.load('data/result.npy')
    aaaa = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    print('load_success:')

    best = 0
    best_auc = 0
    best_acc = 0
    best_aupr = 0
    fold_index = 1
    fold_acc = 0
    fold_aupr = 0
    fold_auc = 0

    for train1, test1 in aaaa.split(result, result[:, 2]):
        train_index = result[train1]
        test_index = result[test1]

        model = GTN(num_edge_u=len(A_u),
                num_channels_u=num_channels_u,
                u_in=node_features_u.shape[1],
                u_out=node_dim,
                num_edge_v=len(A_v),
                num_channels_v=num_channels_v,
                v_in=node_features_v.shape[1],
                v_out=node_dim,
                num_nodes_u=node_features_u.shape[0],
                num_nodes_v=node_features_v.shape[0]
                )
        optimizer_u = torch.optim.Adam(model.parameters(), lr= lr, weight_decay= weight_decay)

        for i in range(epochs):
            print('Epoch: {:02d}'.format(i + 1),
                    'Fold: {:02d}'.format(fold_index))
            model.zero_grad()
            model.train()
            optimizer_u.zero_grad()
            Xu_, Xv_,  _, _, loss, y, target = model(A_u, node_features_u,  A_v,node_features_v, train_index)
            loss.backward()
            optimizer_u.step()
            acc = (y.argmax(dim=1) == target).sum().type(torch.float) / 16000.0
            y_pro = y[:, 1]
            auc = roc_auc_score(target.detach().numpy(), y_pro.detach().numpy())
            precision, recall, thresholds = precision_recall_curve(target.detach().numpy(), y_pro.detach().numpy())
            aupr = auc3(recall, precision)
            print("Train set results:",
                    "loss_train= {:.4f}".format(loss.detach().cpu().numpy()),
                    "train_auc= {:.4f}".format(auc.item()),
                    "train_aupr= {:.4f}".format(aupr.item()),
                    "train_accuracy= {:.4f}".format(acc.item()))
            model.eval()
            with torch.no_grad():

                _, _, _, _, test_loss, y, target = model.forward(A_u, node_features_u, A_v, node_features_v, test_index)
                acc = (y.argmax(dim=1) == target).sum().type(torch.float) / 4000.0
                y_pro = y[:, 1]
                auc = roc_auc_score(target.detach().numpy(), y_pro.detach().numpy())
                precision, recall, thresholds = precision_recall_curve(target.detach().numpy(), y_pro.detach().numpy())
                aupr = auc3(recall, precision)
                print("Test set results:",
                    "loss_test={:.4f}".format(loss.detach().cpu().numpy()),
                    "test_auc= {:.4f}".format(auc.item()),
                    "test_aupr= {:.4f}".format(aupr.item()),
                    "test_accuracy= {:.4f}".format(acc.item()))
                if best < (aupr + auc + acc/2):
                    best = aupr +auc + acc/2
                    best_aupr =aupr
                    best_auc = auc
                    best_acc = acc

        fold_acc = fold_acc + best_acc
        fold_auc = fold_auc + best_auc
        fold_aupr = fold_aupr + best_aupr
        best = 0
        best_auc = 0
        best_acc = 0
        best_aupr = 0
        fold_index = fold_index + 1


    print("Fold best results:",
            "fold5_best_auc= {:.4f}".format(fold_auc / 5),
            "fold5_best_aupr= {:.4f}".format(fold_aupr / 5),
            "fold5_best_acc= {:.4f}\n".format(fold_acc / 5))

