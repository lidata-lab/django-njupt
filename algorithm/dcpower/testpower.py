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
minmax_list = [[0,0,0,0,0,0], [400,200,100000000,100000000,500000000,500000000]]
model_list = [Lasso(), Ridge(), ElasticNet(), BayesianRidge(), SGDRegressor(), Perceptron()]

def data_preprocess(df_list, power_df=None):
    if not isinstance(df_list, list):
        df_list = [df_list]
    print(df_list)
    df_len_list = list(map(lambda x: x.shape[0], df_list))
    print(df_len_list)
    for df in df_list:
        df.iloc[:, -2:] = df.iloc[:, -2:].fillna(0)
    df_list = list(map(lambda x: x.dropna(), df_list))
    df_len_list = list(map(lambda x: x.shape[0], df_list))
    print(df_len_list)
    #print(df_list, '\n', df_len_list)
    if len(df_list) > 1:
    #if min(df_len_list) != max(df_len_list):
        for i, df in enumerate(df_list):
            if i == 0:
                df_index = df.index
            else:
                df_index = df.index.intersection(df_index)
        if isinstance(power_df, pd.DataFrame):
            df_index = power_df.index.intersection(df_index)
        if df_index.empty:
            #print('Empty:\n', df_list)
            df_list = list(map(lambda x: x.iloc[-1:, :], df_list))
            #print('-1\n', df_list)
            if isinstance(power_df, pd.DataFrame):
                power_df = power_df.iloc[-1] - power_baseline
        else:
            df_list = list(map(lambda x: x.loc[df_index, :], df_list))
            if isinstance(power_df, pd.DataFrame):
                power_df = power_df.loc[df_index] - power_baseline
    #df_list = list(map(lambda x: x.fillna(x.median()), df_list))
    if isinstance(power_df, pd.DataFrame):
        return df_list, power_df
    else:
        print('Return:', df_list)
        return df_list

def getmax(df_list):
    df_list = data_preprocess(df_list)
    df_sum = vmsum2one(df_list)
    df_max = df_sum.max()
    max_list = df_max.values.tolist()
    print(max_list)

def vmsum2one(df_list):
    for i, df in enumerate(df_list):
        if i == 0:
            df_sum = df
        else:
            df_sum += df
    print('VM Sum:\n', df_sum)
    return df_sum

def gridsearch_lasso_best(X, y):
    score_best = 0
    param_best = {'alpha': 1}
    for alp in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
        lasso = Lasso(alpha=alp)
        split = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        score = cv(lasso, X, y, cv=split).mean()
        if score > score_best:
            score_best = score
            param_best = {'alpha': alp}
        print(param_best)
    return param_best

def lasso_vm2pc_train(df_list, power_df):
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
        model = RFR()
        X_train, X_test = X_minmax[train_index, :], X_minmax[test_index, :]
        y_train, y_test = y_np[train_index], y_np[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        nrmse_tmp = nrmse(y_test, y_pred)
        if nrmse_tmp < nrmse_best:
            #mfile = open('/home/fc10382/Mcoder/Django/algorithm/dcpower/lmodel/asso.pkl', 'wb')
            mfile = open('dcpower/model/rfr.pkl', 'wb')
            pickle.dump(model, mfile)
            mfile.close()
        #print(model.coef_, model.intercept_, 'NRMSE:', nrmse_tmp)
        print(model.feature_importances_, 'NRMSE:', nrmse_tmp)
        #for model in model_list:
        #    model.fit(X_train, y_train)
        #    y_pred = model.predict(X_test)
        #    nrmse_tmp = nrmse(y_test, y_pred)
        #    if nrmse_tmp < nrmse_best:
        #        mfile = open('dcpower/model/best_linear.pkl', 'wb')
        #        pickle.dump(model, mfile)
        #        mfile.close()
        #    print(model.coef_, model.intercept_, 'NRMSE:', nrmse_tmp)

def lasso_vm2pc_pred(df_list, power_df=None):
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
    #X_minmax = df_sum.values
    #mfile = open('dcpower/model/lasso.pkl', 'rb')
    mfile = open('dcpower/model/rfr.pkl', 'rb')
    #mfile = open('dcpower/model/best_linear.pkl', 'rb')
    lasso = pickle.load(mfile)
    mfile.close()
    y_pred = lasso.predict(X_minmax)
    if isinstance(power_df, pd.DataFrame):
        y_pred_df = pd.DataFrame(y_pred, index=power_df.index, columns=['pdu.pred'])
        y_df = pd.concat([power_df+power_baseline, y_pred_df+power_baseline], axis=1)
        print(y_df)
        y_json = y_df.to_json(orient='columns')
        power_list = json.loads(y_json)
        #print(power_list['pdu.power'])
        return [power_list['pdu.power'], power_list['pdu.pred']]
    else:
        y_pred_df = pd.DataFrame(y_pred, index=df_sum.index, columns=['pdu.pred']) + power_baseline
        y_json = y_pred_df.to_json(orient='columns')
        power_list = json.loads(y_json)
        return [power_list['pdu.pred']]
