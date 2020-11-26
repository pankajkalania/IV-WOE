# WOE and IV in python from scratch
Code in python to calculate WOE and IV from scratch.

## How to use it?
**Step-1 :** Load your data with your binary target feature in a pandas DataFrame.

```{python}
data=pd.read_csv(os.path.join(data_path, "data.csv"))
print(data.shape)
```

**Step-2 :** Call function get_iv_woe() in iv_woe_code.py to get IV and WOE values.

```{python}
iv, woe_iv = get_iv_woe(data.copy(), target_col="bad_customer", max_bins=20)
print(iv.shape, woe_iv.shape)
```
**Note** : Make sure dtype of continuous columns in dataframe is not object. Because it will consider it as categorical and binning won't be done for that column. <br>
Where,
* **iv** DataFrame contains aggregated information values corresponding to every independent feature and some additional information like: binning technique used for feature and null percentage.
* **woe_iv** DataFrame contains bins, their corresponding distributions, WOE and IV values.
