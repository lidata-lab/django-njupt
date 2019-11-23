import numpy as np
import pandas as pd
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold as kflod

from dcpower.testpower import data_preprocess, vmsum2one
from mlv82.utils import to_numpy
from mlv82.metrics.regression import root_mean_squared_error as rmse
from mlv82.metrics.regression import norm_root_mean_squared_error as nrmse

power_baseline = 16
minmax_list = [[0,0,0,0,0,0], [400,200,100000000,100000000,500000000,500000000]]
LR = 0.01
batch = 10
EPOCH = 100
# target_cols = ['ObjectD1_1_FeatureA', 'ObjectB1_FeatureA']
# target_cols = ['ObjectB1_FeatureA']

class linear_model():
    w = torch.tensor([[0.3, 0.1, 0, 0, 0, 0]], dtype=torch.float, requires_grad=True)
    b = torch.tensor([.0], dtype=torch.float, requires_grad=True)
    def __init__(self):
        #self.w = w
        #self.b = b
        pass

    def run(self, x):
        return x @ self.w.t() + self.b

def mse(true, pred):
    diff = pred - true
    return torch.sum(diff * diff) / diff.numel()

def mse_penalty(true, pred):
    diff = pred - true
    beyond = torch.max(torch.zeros(diff.numel(), diff))
    return mse(true, pred) + 1000 * torch.sum(beyond * beyond) / diff.numel()

class vm_net_1d(torch.nn.Module):
    def __init__(self, in_num, out_num):
        super(vm_net_1d, self).__init__()
        self.linear = nn.Linear(in_num, out_num)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        out = self.linear(x)
        return out

class vm_net(torch.nn.Module):
    def __init__(self, in_num, out_num):
        super(vm_net, self).__init__()
        self.linear1 = nn.Linear(in_num, 15)
        self.linear2 = nn.Linear(15, 10)
        self.linear3 = nn.Linear(10, out_num)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        return out

class VmLoss(nn.Module):
    def __init__(self, w, a):
        super(VmLoss, self).__init__()
        self.w = w
        self.a = a
        return

    def forward(self, true, pred):
        square_error = torch.mean(torch.pow(pred-true, 2))
        penalty = torch.mean(torch.pow(torch.max(torch.zeros(true.size()[0], dtype=pred.dtype).cuda(), pred-true), 2))
        loss = self.w * square_error + self.a * penalty
        return loss

def nn_toy(df_list, power_df):
    df_list, power_df = data_preprocess(df_list, power_df)
    df_sum = vmsum2one(df_list)
    minmax_scaler = MinMaxScaler()
    # minmax_scaler.fit(minmax_list)
    # X_minmax = minmax_scaler.transform(df_sum)
    X_minmax = minmax_scaler.fit_transform(df_sum)
    y_np = power_df.values.astype(np.float)
    X_train = X_minmax[:-200, :]
    y_train = y_np[:-200]
    X_test = X_minmax[-200:, :]
    y_test = y_np[-200:]
    X_minmax_ts = torch.tensor(X_train, dtype=torch.float, requires_grad=True)
    y_ts = torch.tensor(y_train, dtype=torch.float)
    X_test_ts = torch.tensor(X_test, dtype=torch.float, requires_grad=True)
    y_test_ts = torch.tensor(y_test, dtype=torch.float)
    data_ds = TensorDataset(X_minmax_ts, y_ts)
    data_dl = DataLoader(data_ds, batch, shuffle=True)
    # model = linear_model()
    model = nn.Linear(6, 1)
    torch.nn.init.xavier_uniform_(model.weight)
    print(model.weight)
    print(model.bias)
    loss_fun = VmLoss(1, 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCH):
        for x, y in data_dl:
            pred = model(x)
            loss = loss_fun(y, pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 10 == 0:
            print(loss.detach().numpy())
            print(model.weight, model.bias)
            print(model.weight.grad, model.bias.grad)
    y_pred_ts = model(X_test_ts)
    nrmse_val = nrmse(y_test, y_pred_ts.detach().numpy())
    print(nrmse_val)
    return y_test, y_pred_ts.detach().numpy()

def nn_vm2pc_train(df_list, power_df):
    df_list, power_df = data_preprocess(df_list, power_df)
    df_sum = vmsum2one(df_list)
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(minmax_list)
    X_minmax = minmax_scaler.transform(df_sum)
    #print(X_minmax)
    # X_minmax = df_sum.values
    y_np = power_df.values
    #param_best = gridsearch_lasso_best(X_minmax, y_np)['alpha']
    nrmse_best = 100
    ssplit = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train_index, test_index in ssplit.split(X_minmax, y_np):
        #model = Lasso(alpha=param_best)
        X_train, X_test = X_minmax[train_index, :], X_minmax[test_index, :]
        y_train, y_test = y_np[train_index], y_np[test_index]
        X_train_ts = torch.tensor(X_train).cuda()
        y_train_ts = torch.tensor(y_train).double().cuda()
        net = vm_net_1d(6, 1).double().cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        # loss_fun = VmLoss(1, 1000)
        loss_fun = torch.nn.MSELoss()
        for t in range(10000):
            y_train_pred_ts = net(X_train_ts)
            loss = loss_fun(y_train_ts, y_train_pred_ts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 100 == 0:
                print(loss.cpu().detach().numpy())
        X_test_ts = torch.tensor(X_test).cuda()
        y_test_ts = torch.tensor(y_test).cuda()
        y_pred_ts = net(X_test_ts).double()
        nrmse_tmp = nrmse(y_test, y_pred_ts.cpu().detach().numpy())
        if nrmse_tmp < nrmse_best:
            #mfile = open('/home/fc10382/Mcoder/Django/algorithm/dcpower/lasso.pkl', 'wb')
            print(net.linear.weight)
            print('NRMSE:', nrmse_tmp)
            nrmse_best = nrmse_tmp
            torch.save(net, '/home/fc10382/Mcoder/Django/algorithm/dcpower/linear1.pkl')
            #torch.save(net, '/home/fc10382/Mcoder/Django/algorithm/dcpower/nn3.pkl')
        #print(model.coef_, model.intercept_, 'NRMSE:', nrmse_tmp)
        

def nn_vm2pc_pred(df_list, power_df=None):
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(minmax_list)
    print(type(power_df))
    if isinstance(power_df, pd.DataFrame):
        df_list, power_df = data_preprocess(df_list, power_df)
    else:
        df_list = data_preprocess(df_list)
    if len(df_list) == 1:
        df_sum = df_list[0]
    else:
        df_sum = vmsum2one(df_list)
    X_minmax = minmax_scaler.transform(df_sum)
    X_minmax_ts = torch.tensor(X_minmax).cuda()
    net = torch.load('/home/fc10382/Mcoder/Django/algorithm/dcpower/linear1.pkl')
    y_pred_ts = net(X_minmax_ts)
    if isinstance(power_df, pd.DataFrame):
        y_pred_df = pd.DataFrame(y_pred_ts.cpu().detach().numpy(), index=power_df.index, columns=['pdu.pred'])
        y_df = pd.concat([power_df+power_baseline, y_pred_df+power_baseline], axis=1)
        print(y_df)
        y_json = y_df.to_json(orient='columns')
        power_list = json.loads(y_json)
        #print(power_list['pdu.power'])
        return [power_list['pdu.power'], power_list['pdu.pred']]
    else:
        y_pred_df = pd.DataFrame(y_pred_ts.cpu().detach().numpy(), index=df_sum.index, columns=['pdu.pred']) + power_baseline
        y_json = y_pred_df.to_json(orient='columns')
        power_list = json.loads(y_json)
        return [power_list['pdu.pred']]
