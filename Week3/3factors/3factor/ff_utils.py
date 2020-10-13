# -*- coding: utf-8 -*-
"""
Created on Fri Oct 2 08:25:53 2020

@author: zheng
"""
import pandas as pd

def record_results(results, data, feature_name, port_name):
    temp_df = pd.DataFrame(data=[data], columns=['a', 'b', 's', 'h'])
    temp_df['port'] = port_name
    temp_df['feature'] = feature_name
    results = results.append(temp_df)
    return results

def get_ff_format(results):
    results['ME'] = results['port'].apply(lambda x: x.split(" ")[0])
    results['BM'] = results['port'].apply(lambda x: x.split(" ")[1])
    results = results.pivot_table(index=['feature','BM'], columns=['ME'], values=['b','s','h'])
    results = results.T
    return results
    