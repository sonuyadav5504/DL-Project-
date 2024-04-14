import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import os

class GraphRec(nn.Module):
    def __init__(self , num_users , num_items , num_ratings , history_u , history_i , history_ur ,\
                                     history_ir , embed_dim , social_neighbor , cuda = 'cpu'):
        super(GraphRec, self).__init__()
        self.embed_dim = embed_dim
        u2e = nn.Embedding(num_users , self.embed_dim)
        i2e = nn.Embedding(num_items , self.embed_dim)
        r2e = nn.Embedding(num_ratings , self.embed_dim)
        self.enc_u = ui_aggregator(i2e , r2e , u2e , embed_dim , history_u , history_ur , cuda , user=True)
        self.enc_i = ui_aggregator(i2e , r2e , u2e , embed_dim , history_i , history_ir , cuda , user=False)
        self.enc_social = social_aggregator(None , u2e , embed_dim , social_neighbor , cuda)

        self.w_u = nn.Linear(2*self.embed_dim , self.embed_dim)
        self.w_ur1 = nn.Linear(self.embed_dim , self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim , self.embed_dim)
        self.w_ir1 = nn.Linear(self.embed_dim , self.embed_dim)
        self.w_ir2 = nn.Linear(self.embed_dim , self.embed_dim)
        self.w_ui1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_ui2 = nn.Linear(self.embed_dim , 16)
        self.w_ui3 = nn.Linear(16 , 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim , momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim , momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim , momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16 , momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self , nodes_u , nodes_i):
        item_space = self.enc_u(nodes_u)
        social_space = self.enc_social(nodes_u)
        user_latent_feature = torch.cat([item_space , social_space], dim=1)
        user_latent_feature = F.relu(self.w_u(user_latent_feature))
        item_latent_feature = self.enc_i(nodes_i)

        user_latent_feature = F.relu(self.bn1(self.w_ur1(user_latent_feature)))
        user_latent_feature = F.dropout(user_latent_feature, training=self.training)
        user_latent_feature = self.w_ur2(user_latent_feature)
        item_latent_feature = F.relu(self.bn2(self.w_ir1(item_latent_feature)))
        item_latent_feature = F.dropout(item_latent_feature , training=self.training)
        item_latent_feature = self.w_ir2(item_latent_feature)

        latent_feature = torch.cat((user_latent_feature , item_latent_feature), 1)
        latent_feature = F.relu(self.bn3(self.w_ui1(latent_feature)))
        latent_feature = F.dropout(latent_feature , training=self.training)
        latent_feature = F.relu(self.bn4(self.w_ui2(latent_feature)))
        latent_feature = F.dropout(latent_feature , training=self.training)
        scores = self.w_ui3(latent_feature)

        return scores.squeeze()

    def loss(self , nodes_u , nodes_i , ratings):
        scores = self.forward(nodes_u , nodes_i)
        return self.criterion(scores , ratings)

class Attention(nn.Module):
    def __init__(self , embedding_dims):
        super(Attention , self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim , self.embed_dim , 1)
        self.att1 = nn.Linear(self.embed_dim * 2 , self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim , self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim , 1)
        self.softmax = nn.Softmax(0)

    def forward(self , node1 , u_rep , num_neighs):
        uv_reps = u_rep.repeat(num_neighs , 1)
        x = torch.cat((node1 , uv_reps) , 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x , training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x , training=self.training)
        x = self.att3(x)
        att = F.softmax(x , dim=0)
        return att

class ui_aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """
    def __init__(self , i2e , r2e , u2e , embed_dim , history_ui , history_r , cuda = "cpu" , user = True):
        super(ui_aggregator, self).__init__()
        self.user = user
        self.i2e = i2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.history_ui = history_ui
        self.history_r = history_r
        self.w_r1 = nn.Linear(self.embed_dim*2 , self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim , self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.linear1 = nn.Linear(2*self.embed_dim , self.embed_dim) 

    def forward(self, nodes):
        ui_history = []
        r_history = []
        for node in nodes:
            ui_history.append(self.history_ui[int(node)])
            r_history.append(self.history_r[int(node)])

        num_len = len(ui_history)
        embed_matrix = torch.empty(num_len , self.embed_dim , dtype = torch.float).to(self.device)

        for i in range(num_len):
            history = ui_history[i]
            num_histroy_ui = len(history)
            tmp_label = r_history[i]

            if self.user == True:
                # user component
                e_ui = self.i2e.weight[history]
                ui_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_ui = self.u2e.weight[history]
                ui_rep = self.i2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_ui , e_r) , 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history , ui_rep , num_histroy_ui)
            att_history = torch.mm(o_history.t() , att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        neigh_feats = embed_matrix

        if self.user == True:
            self_feats = self.u2e.weight[nodes]
        else:
            self_feats = self.i2e.weight[nodes]
        combined = torch.cat([self_feats , neigh_feats] , dim = 1)
        combined_feats = F.relu(self.linear1(combined))
        return combined_feats

class social_aggregator(nn.Module):
    """
    social aggregator: for aggregating embeddings of social neighbors.
    """
    def __init__(self , features , u2e , embed_dim , social_neighbor , cuda = "cpu"):
        super(social_aggregator , self).__init__()
        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.social_neighbor = social_neighbor
        self.att = Attention(self.embed_dim)
        self.linear1 = nn.Linear(2*self.embed_dim , self.embed_dim)

    def forward(self, nodes):
        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_neighbor[int(node)])
        num_len = len(nodes)
        embed_matrix = torch.empty(num_len , self.embed_dim , dtype = torch.float).to(self.device)
        for i in range(num_len):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            e_u = self.u2e.weight[list(tmp_adj)] # user embedding 

            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u , u_rep , num_neighs)
            att_history = torch.mm(e_u.t() , att_w).t()
            embed_matrix[i] = att_history
        neigh_feats = embed_matrix

        self_feats = self.u2e.weight[nodes]
        combined = torch.cat([self_feats , neigh_feats] , dim = 1)
        combined_feats = F.relu(self.linear1(combined))

        return combined_feats


############################################ run_graphrec ###########################################
def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        batch_nodes_u, batch_nodes_i, batch_ratings = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_i.to(device), batch_ratings.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 60 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 60, best_rmse, best_mae))
            running_loss = 0.0

    return 0

def test(model, device, test_loader):
    model.eval()
    pred = []
    target = []
    with torch.no_grad():
        for test_u, test_i, test_ratings in test_loader:
            test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
            scores = model(test_u, test_i)
            pred.append(list(scores.cpu().numpy()))
            target.append(list(test_ratings.cpu().numpy()))
    pred = np.array(sum(pred, []))
    target = np.array(sum(target, []))
    rmse = sqrt(mean_squared_error(pred, target))
    mae = mean_absolute_error(pred, target)

    return rmse, mae


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = 128

    path_data = "dataset.pkl"
    data_file = open(path_data, 'rb')
    history_u, history_i, history_ur, history_ir, train_u, train_i, train_r,\
                 test_u, test_i, test_r, social_neighbor, ratings = pickle.load(data_file)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i),
                                              torch.FloatTensor(train_r))
    
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_i),
                                             torch.FloatTensor(test_r))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
    num_users = history_u.__len__()
    num_items = history_i.__len__()
    num_ratings = ratings.__len__()

    # model
    graphrec = GraphRec(num_users, num_items, num_ratings, history_u, history_i, history_ur,\
                                     history_ir, embed_dim, social_neighbor, cuda=device).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=0.01, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
 
    for epoch in range(20):

        train(graphrec, device, train_loader, optimizer, epoch+1, best_rmse, best_mae)
        expected_rmse, mae = test(graphrec, device, test_loader)

        # early stopping
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break
    test_rmse, test_mae = test(graphrec, device, test_loader)   
    print("test_rmse: %.4f, test_mae:%.4f " % (test_rmse, test_mae))

if __name__ == "__main__":
    main()
