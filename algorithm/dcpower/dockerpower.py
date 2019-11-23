import numpy as np
import pandas as pd
import pickle
import random
from dcpower.testpower import data_preprocess
from mlv82.utils import to_numpy
from sklearn.preprocessing import MinMaxScaler
from mlv82.metrics.regression import root_mean_squared_error as rmse
from mlv82.metrics.regression import norm_root_mean_squared_error as nrmse

metrics_suffix = ['cpu.percent', 'memory.percent', 'network.usage.rx_bytes', 'network.usage.tx_bytes', 'blkio.io_service_bytes_recursive-253-0.write', 'blkio.io_service_bytes_recursive-253-0.read']
minmax_list = [[0,0,0,0,0,0], [400,200,100000000,100000000,500000000,500000000]]

def build_docker_metrics(vmno, dockernum):
    docker_metrics_list = []
    for i in range(dockernum):
        docker_metrics_list.append(list(map(lambda x: 'docker.linpack%s%d.%s' % (vmno, i, x), metrics_suffix)))
    return docker_metrics_list

def calc_docker_power(df, vmno, dockerno=None):
    df_list = data_preprocess(df)
    #data_df = df.dropna()
    # mfile = open('dcpower/model/lasso.pkl', 'rb')
    mfile = open('dcpower/model/rfr.pkl', 'rb')
    lasso = pickle.load(mfile)
    mfile.close()
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(minmax_list)
    y_pred_mean_list = []
    for data_df in df_list:
        X_minmax = minmax_scaler.transform(data_df) 
        y_pred = lasso.predict(X_minmax)
        y_pred_mean_list.append(pd.DataFrame(y_pred).mean().values[0])
        print('y_pred: ', y_pred)
    power_sum = sum(y_pred_mean_list[-2:])
    print(y_pred_mean_list)
    if power_sum > y_pred_mean_list[0]:
        percent_1 = y_pred_mean_list[-2] / power_sum + random.uniform(-0.1, 0.1)
        y_pred_mean_list[-2] = y_pred_mean_list[0] * percent_1
        y_pred_mean_list[-1] = y_pred_mean_list[0] * (1 - percent_1)
    return y_pred_mean_list
