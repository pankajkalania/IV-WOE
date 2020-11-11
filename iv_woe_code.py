#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np, os, re, math, time

# to check monotonicity of a series
def is_monotonic(temp_series):
    return all(temp_series[i] <= temp_series[i + 1] for i in range(len(temp_series) - 1)) or all(temp_series[i] >= temp_series[i + 1] for i in range(len(temp_series) - 1))

def prepare_bins(bin_data, c_i, target_col, max_bins):
    force_bin = True
    binned = False
    remarks = np.nan
    # ----------------- Monotonic binning -----------------
    for n_bins in range(max_bins, 2, -1):
        try:
            bin_data[c_i + "_bins"] = pd.qcut(bin_data[c_i], n_bins).astype(object)
            if is_monotonic(bin_data.groupby(c_i + "_bins")[target_col].mean().reset_index(drop=True)):
                force_bin = False
                binned = True
                remarks = "binned monotonically"
                break
        except:
            pass
    # ----------------- Force binning -----------------
    # creating 2 bins forcefully because 2 bins will always be monotonic
    if force_bin:
        bin_data[c_i + "_bins"] = pd.qcut(bin_data[c_i], 2, duplicates='drop').astype(object)
        if bin_data[c_i + "_bins"].nunique() == 2:
            binned = True
            remarks = "binned forcefully"

    if binned:
        return c_i + "_bins", remarks, bin_data[[c_i+"_bins", target_col]].copy()
    else:
        remarks = "couldn't bin"
        return c_i, remarks, bin_data[[c_i, target_col]].copy()


# calculate WOE and IV for every group/bin/class for a provided feature
def iv_woe_4iter(binned_data, target_col, class_col):
    binned_data = binned_data.fillna("Missing")
    temp_groupby = binned_data.groupby(class_col)[target_col].agg(["count", lambda x: (x == 0).sum(), lambda x: (x == 1).sum()])
    temp_groupby = temp_groupby.reset_index()
    temp_groupby.columns = ["sample_class", "sample_count", "good_count", "bad_count"]
    temp_groupby["feature"] = class_col
    if "_bins" in class_col:
        temp_groupby["sample_class_label"]=temp_groupby["sample_class"].replace({"Missing": np.nan}).astype('category').cat.codes.replace({-1: np.nan})
    else:
        temp_groupby["sample_class_label"]=np.nan
    temp_groupby = temp_groupby[["feature", "sample_class", "sample_class_label", "sample_count", "good_count", "bad_count"]]
    
    """
    **********get distribution of good and bad
    Note: distribution formulae is adjusted for classes where good_count or bad_count is 0.
    """
    temp_groupby['distbn_good'] = temp_groupby.apply(lambda x: x["good_count"]/temp_groupby['good_count'].sum() if x["good_count"] > 0 else (x["good_count"] + 0.5)/temp_groupby['good_count'].sum(), axis=1)
    temp_groupby['distbn_bad'] = temp_groupby.apply(lambda x: x["bad_count"]/temp_groupby['bad_count'].sum() if x["bad_count"] > 0 else (x["bad_count"] + 0.5)/temp_groupby['bad_count'].sum(), axis=1)

    temp_groupby['woe'] = np.log(temp_groupby['distbn_good'] / temp_groupby['distbn_bad'])
    temp_groupby['iv'] = (temp_groupby['distbn_good'] - temp_groupby['distbn_bad']) * temp_groupby['woe']
    
    return temp_groupby

"""
- iterate over all features.
- calculate WOE & IV for there classes.
- append to one DataFrame woe_iv.
"""
def var_iter(data, target_col, max_bins):
    woe_iv = pd.DataFrame()
    remarks_list = []
    for c_i in data.columns:
        if c_i not in [target_col]:
            # check if binning is required. if yes, then prepare bins and calculate woe and iv.
            if np.issubdtype(data[c_i], np.number) and data[c_i].nunique() > 2:
                class_col, remarks, binned_data = prepare_bins(data[[c_i, target_col]].copy(), c_i, target_col, max_bins)
                agg_data = iv_woe_4iter(binned_data.copy(), target_col, class_col)
                remarks_list.append({"feature": c_i, "remarks": remarks})
            else:
                agg_data = iv_woe_4iter(data[[c_i, target_col]].copy(), target_col, c_i)
                remarks_list.append({"feature": c_i, "remarks": np.nan})
            woe_iv = woe_iv.append(agg_data)
    return woe_iv, pd.DataFrame(remarks_list)

# after getting woe and iv for all classes of features calculate aggregated IV values for features.
def get_iv_woe(data, target_col, max_bins, fill_by_woe=False, woe_var_list=[]):
    func_start_time = time.time()
    woe_iv, binning_remarks = var_iter(data, target_col, max_bins)
    woe_iv["sample_class_min"] = woe_iv["sample_class"].apply(lambda x:x.left if type(x) == pd._libs.interval.Interval else x)
    woe_iv["sample_class_max"] = woe_iv["sample_class"].apply(lambda x:x.right if type(x) == pd._libs.interval.Interval else x)
    
    woe_iv["feature"] = woe_iv["feature"].replace("_bins", "", regex=True)
    woe_iv = woe_iv[['feature', 'sample_class', 'sample_class_label', 'sample_class_min', 'sample_class_max',
                     'sample_count', 'good_count', 'bad_count', 'distbn_good', 'distbn_bad', 'woe', 'iv']]
    
    iv = woe_iv.groupby("feature")[["iv"]].agg(["sum", "count"]).reset_index()
    iv.columns = ["feature", "iv", "number_of_classes"]
    iv["feature_null_percent"] = iv["feature"].apply(lambda x:data.isnull().mean()[x])
    iv = iv.merge(binning_remarks, on="feature", how="left")
    
    print("Total time elapsed: {} minutes".format(round((time.time() - func_start_time) / 60, 3)))
    return iv, woe_iv.replace({"Missing": np.nan})
