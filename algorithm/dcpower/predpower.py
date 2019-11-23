import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge, SGDRegressor, Perceptron
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import ShuffleSplit, check_cv
from sklearn.model_selection import cross_val_score as cv
from mlv82.utils import to_numpy
from mlv82.metrics.regression import root_mean_squared_error as rmse
from mlv82.metrics.regression import norm_root_mean_squared_error as nrmse

power_baseline = 16
minmax_list = [[66.382883310318, 59.08784103393555, 20676.8125, 1586479.5625, 2269024.0, 9350656.0, 59.0] * 30, [51.62166404724121, 58.498998403549194, 4430.8125, 82720.25, 35360.0, 0.0, 44.8125] * 30]
model_list = [Lasso(), Ridge(), ElasticNet(), BayesianRidge(), SGDRegressor(), Perceptron()]

def time_mean(data_df, time_unit):
    ts_start = int(data_df.index[0])
    ts_end = int(data_df.index[-1])
    ts_tmp = ts_start
    for i in range(int((ts_end - ts_start) / time_unit) + 1):
        if i == 0:
            mean_df = pd.DataFrame(data_df.loc[str(ts_tmp):str(ts_tmp+time_unit)].mean(axis=0, skipna=True)).T
        else:
            mean_df = pd.concat([mean_df, pd.DataFrame(data_df.loc[str(ts_tmp):str(ts_tmp+time_unit)].mean(axis=0, skipna=True)).T], ignore_index=True)
        ts_tmp += time_unit
    return mean_df

def fea_rebuild(fea_df, power_df, time_len=30, train=0):
    new_df = pd.concat([fea_df, power_df], axis=1)
    print(train)
    if train == 0:
        #print(new_df)
        new_fea_np = np.expand_dims(new_df.values.flatten(), axis=0)
        print(new_fea_np.shape)
        return new_fea_np
    else:
        for i in range(new_df.shape[0] - time_len):
            if i == 0:
                new_fea_np = new_df.iloc[i:i+time_len,:].values.flatten()
            else:
                new_fea_np = np.vstack((new_fea_np, new_df.iloc[i:i+time_len,:].values.flatten()))
        y_np = power_df.iloc[time_len:].values.squeeze()
        print(new_fea_np.shape, y_np.shape)
        return new_fea_np, y_np

def pred_preprocess(his_df, power_df, time_unit, train=0):
    mean_his_df = time_mean(his_df, time_unit)
    mean_power_df = time_mean(power_df, time_unit)
    print(mean_his_df.shape)
    print(mean_power_df.shape)
    if train == 0:
        X_np = fea_rebuild(mean_his_df, mean_power_df, train=train)
        return X_np, mean_power_df.values
    else:
        X_np, y_np = fea_rebuild(mean_his_df, mean_power_df, train=train)
        return X_np, y_np

def pcpower_pred_train(df_list, power_df, time_unit):
    X_np, y_np = pred_preprocess(df_list, power_df, time_unit, train=1)
    #print(list(X_np.max(axis=0)))
    #print(list(X_np.min(axis=0)))
    #return 0
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(minmax_list)
    X_minmax = minmax_scaler.transform(X_np)
    #print(X_minmax)
    nrmse_best = 1000
    ssplit = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train_index, test_index in ssplit.split(X_minmax, y_np):
        #model = Lasso(alpha=param_best)
        model = RFR()
        X_train, X_test = X_minmax[train_index, :], X_minmax[test_index, :]
        y_train, y_test = y_np[train_index], y_np[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        nrmse_tmp = nrmse(y_test, y_pred)
        if nrmse_tmp < nrmse_best:
            #mfile = open('/home/fc10382/Mcoder/Django/algorithm/dcpower/lasso.pkl', 'wb')
            if time_unit == 15:
                mfile = open('dcpower/model/pred_rfr.pkl', 'wb')
            elif time_unit == 10:
                mfile = open('dcpower/model/pred_rfr-10.pkl', 'wb')
            elif time_unit == 5:
                mfile = open('dcpower/model/pred_rfr-5.pkl', 'wb')
            pickle.dump(model, mfile)
            mfile.close()
            nrmse_best = nrmse_tmp
        #print(model.coef_, model.intercept_, 'NRMSE:', nrmse_tmp)
        print(model.feature_importances_, 'NRMSE:', nrmse_tmp)

def pcpower_pred(df_list, power_df, time_unit):
    X_np, y_np = pred_preprocess(df_list, power_df, time_unit)
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(minmax_list)
    X_minmax = minmax_scaler.transform(X_np)
    #X_minmax = df_sum.values
    #mfile = open('/home/fc10382/Mcoder/Django/algorithm/dcpower/lasso.pkl', 'rb')
    if time_unit == 15:
        mfile = open('dcpower/model/pred_rfr.pkl', 'rb')
    elif time_unit == 10:
        mfile = open('dcpower/model/pred_rfr-10.pkl', 'rb')
    elif time_unit == 5:
        mfile = open('dcpower/model/pred_rfr-5.pkl', 'rb')
    #mfile = open('/home/fc10382/Mcoder/Django/algorithm/dcpower/best_linear.pkl', 'rb')
    lasso = pickle.load(mfile)
    mfile.close()
    y_pred = lasso.predict(X_minmax)
    #print(power_df.values)
    #print(y_np)
    return y_np.tolist()[-1][0], y_pred.tolist()[0]
