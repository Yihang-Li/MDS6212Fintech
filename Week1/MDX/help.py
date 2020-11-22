# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:55:15 2020

@author: Administrator
"""

# 基础的东西 这一part貌似挺好滴
from sp_functions import get_data
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_data(name, cols=None, fmt=None, skipfooter=0):
      df = pd.read_csv(name, engine='python',skipfooter=skipfooter)
      df = df.set_index('Date')
      df.index = pd.to_datetime(df.index, format=fmt)
      if cols is None:
            return df
      else:
            return df[cols]
        
data = get_data("data-220020013.csv")
data = data.dropna(how='any')

# 1 用max return 限制weight sum为1 平均年度std为0.2
def cal_risk_contribution(w, cov):
      w = w[:,None]
      sigma_p = np.sqrt(np.matmul(np.matmul(w.T,cov), w))
      risk_contrib = w * np.matmul(cov, w) / sigma_p
      risk_contrib_pct = risk_contrib/sigma_p
      return risk_contrib_pct.squeeze()

def record_results(weight, asset_return,port_val):
      asset_return_cum = (asset_return + 1).cumprod()
      port_return_cum = (asset_return_cum * weight).sum(axis=1)
      port_val_new = port_return_cum * port_val.iloc[-1,0]
      port_val_new = port_val_new.to_frame()
      port_val_new.columns = port_val.columns
      port_val = port_val.append(port_val_new)
      return port_val
  
def cal_return(w):
      w = w[:,None]
      train_data1 = np.array(train_data)
      re = -np.sum(np.matmul(train_data1,w),axis=0)
      return re.squeeze()
  
def get_weight(cov):
      n_assets = len(cov)
      init_w = np.array([1/n_assets] * n_assets)
      def opt(w, cov):
            return_all = cal_return(w)
            return return_all
      cons = [{'type':'eq', 'fun':lambda x:np.sum(x)-1},{'type':'ineq', 'fun':lambda x:-np.sqrt(np.matmul(np.matmul(x.T,cov), x))+0.05}]
      bounds = [[0, 1]]*n_assets
      res = minimize(opt, init_w, args=(cov,),constraints=cons, method='SLSQP',bounds=bounds)
      assert res.success
      return res.x
  # 1）这个里面的cons设置有问题，sum权重是一，但是后面的std不是整个区间的（year=5），而要是一年的，又不可以简单/区间长度（5），因为不是年平均std小于20%是每一年都是小于20%，那是不是要写个函数还是咋样，我不会
rebal_days = ['2014/09/30','2015/09/30','2016/09/30','2017/09/30','2018/09/30','2019/09/30']
rebal_days = [pd.to_datetime(x) for x in rebal_days]

port_value = pd.DataFrame(index=[rebal_days[0]], data=[[100]], columns=["Risk Parity"])
port_value_eq = pd.DataFrame(index=[rebal_days[0]], data=[[100]], columns=["Equal Weight"])
  
for rebal_date, next_date in zip(rebal_days[:-1],rebal_days[1:]):
      train_start_date = rebal_date - pd.DateOffset(years=5)
      train_data = data.loc[train_start_date:rebal_date]
      cov = train_data.cov().values
      w_star = get_weight(cov)
      asset_return = data.loc[rebal_date:next_date]
      port_value = record_results(w_star, asset_return, port_value)
      w_eq = np.array([1/10]*10)
      port_value_eq = record_results(w_eq, asset_return, port_value_eq)

sigma_p_eq = np.sqrt(np.matmul(np.matmul(w_eq.T,cov), w_eq))
sigma_p = np.sqrt(np.matmul(np.matmul(w_star.T,cov), w_star))

# 问题是只配了两只基金。。。
# max 夏普比率
def cal_return2(w,cov):
      w = w[:,None]
      train_data1 = np.array(train_data)
      sigma_p = np.sqrt(np.matmul(np.matmul(w.T,cov), w))
      re = (-1)*((np.sum(np.matmul(train_data1,w),axis=0))/(sigma_p))
      return re.squeeze()
  #2）这里就是我问你的分数怎么处理，我不知道直接这样处理行不行，你说后面的method要改，但是我查了资料也不知道怎么改。
def get_weight2(cov):
      n_assets = len(cov)
      init_w = np.array([1/n_assets] * n_assets)
      def opt(w, cov):
            return_all = cal_return2(w, cov)
            return return_all
      cons = [{'type':'eq', 'fun':lambda x:np.sum(x)-1},{'type':'ineq', 'fun':lambda x:-np.sqrt(np.matmul(np.matmul(x.T,cov), x))+0.05}]
      bounds = [[0, 1]]*n_assets
      res = minimize(opt, init_w, args=(cov,),constraints=cons, method='SLSQP',bounds=bounds)
      assert res.success
      return res.x
  
for rebal_date, next_date in zip(rebal_days[:-1],rebal_days[1:]):
      train_start_date = rebal_date - pd.DateOffset(years=5)
      train_data = data.loc[train_start_date:rebal_date]
      cov = train_data.cov().values
      w_star = get_weight2(cov)
      asset_return = data.loc[rebal_date:next_date]
      port_value2 = record_results(w_star, asset_return, port_value)
      w_eq = np.array([1/10]*10)
      port_value_eq2 = record_results(w_eq, asset_return, port_value_eq)

sigma_p_eq2 = np.sqrt(np.matmul(np.matmul(w_eq.T,cov), w_eq))
sigma_p2 = np.sqrt(np.matmul(np.matmul(w_star.T,cov), w_star))
# 问题：只配了四只并且std超级低

# 后面不用看
def Max_Diversification_Ratio(n ,cov):
    def Get_Diversification_Ratio(w):
       cov_mdr = np.asmatrix(cov)
       temp = np.dot(w,cov_mdr.diagonal().T)
       sigma_p = np.sqrt(np.dot(np.dot(w.T,cov_mdr), w))
       return sigma_p/temp
    init_w = np.array([1/n]*n, ndmin=2)
    bounds = [[0, 1]]*n
    cons = [{'type':'eq', 'fun': lambda x: np.sum(x)-1}]                    
    res = minimize(Get_Diversification_Ratio, init_w, constraints = cons, bounds=bounds)
    return res.x

for rebal_date, next_date in zip(rebal_days[:-1],rebal_days[1:]):
      train_start_date = rebal_date - pd.DateOffset(years=5)
      train_data = data.loc[train_start_date:rebal_date]
      cov = train_data.cov().values
      w_star = Max_Diversification_Ratio(10 ,cov)
      asset_return = data.loc[rebal_date:next_date]
      port_value3 = record_results(w_star, asset_return, port_value)
      w_eq = np.array([1/10]*10)
      port_value_eq3 = record_results(w_eq, asset_return, port_value_eq)

sigma_p_eq3 = np.sqrt(np.matmul(np.matmul(w_eq.T,cov), w_eq))
sigma_p3 = np.sqrt(np.matmul(np.matmul(w_star.T,cov), w_star))

def global_minimum_variance(data, long = 1):
    def GMV_function(w,corr):
        obj_function = np.dot(w.T,np.dot(corr,w))
        return obj_function
    corr = data.corr()
    n = corr.shape[0]
    w = np.ones(n) /n
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    bnds = [(0,0.1) for i in w]
    if long == 1:
        res = minimize(GMV_function, x0 = w, args = (corr), method = 'SLSQP', constraints = cons,bounds = bnds, tol = 1e-30)
    else:
        res = minimize(GMV_function, x0 = w, args = (corr), method = 'SLSQP', constraints = cons, tol = 1e-30)
    return res.x

for rebal_date, next_date in zip(rebal_days[:-1],rebal_days[1:]):
      train_start_date = rebal_date - pd.DateOffset(years=5)
      train_data = data.loc[train_start_date:rebal_date]
      cov = train_data.cov().values
      w_star4 = Max_Diversification_Ratio(10 ,cov)
      asset_return = data.loc[rebal_date:next_date]
      port_value4 = record_results(w_star, asset_return, port_value)
      w_eq = np.array([1/10]*10)
      port_value_eq4 = record_results(w_eq, asset_return, port_value_eq)

sigma_p_eq4 = np.sqrt(np.matmul(np.matmul(w_eq.T,cov), w_eq))
sigma_p4 = np.sqrt(np.matmul(np.matmul(w_star4.T,cov), w_star4))