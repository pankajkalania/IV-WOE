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
iv, woe_iv = get_iv_woe(data.copy(), "bad_customer", 20)
print(iv.shape, woe_iv.shape)
```
