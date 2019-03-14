import pandas as pd

# load dataset
dataset_raw = pd.read_csv('exchange_rates_FED.csv', header=3, index_col=0).iloc[2:,:]

# fix column names
dataset_raw = dataset_raw.add_prefix('USD/')
dataset_raw = dataset_raw.rename(index=str, columns={"USD/USD":"EUR/USD", "USD/USD.1":"GBP/USD", "USD/USD.2":"AUD/USD", "USD/USD.3":"NZD/USD",
                                            "USD/Unnamed: 19":"NBDI", "USD/Unnamed: 20":"NMCDI", "USD/Unnamed: 21":"NOITPI"})

# save index
dates = dataset_raw.index  

# to numeric
prices = pd.DataFrame(columns=dataset_raw.columns)
for col in prices.columns:
    prices[col] = pd.to_numeric(dataset_raw[col].astype(str), errors="coerce")


# ### Dataset overview
# - 12170 rows (days)
#     - start date: January 4, 1971
#     - end date: August 25, 2017
# - 26 columns (23 currencies vs. USD, plus 3 indices*)
#     - series of different length, rest NaN
#     - e.g. EUR/USD starts in 1999
#     - most other main currency pairs in 1971
# 
#     *(Nominal Broad Dollar Index, Nominal Major Currencies Dollar Index, Nominal Other Important Trading Partners Dollar Index)

# #### Create one-day returns
# 
# - create a separate dataframe for daily returns
# - skip `NaN`s instead of padding them with zeroes - that would mislead the models with artificial data

returns = pd.DataFrame(index=prices.index, columns=prices.columns)

for col in returns.columns:
    returns[col] = prices[col][prices[col].notnull()].pct_change()


info = 'Data set: FED Exchange Rates 1971-2017 \
        \nThis module can return the following variables: \
        \ndataset_raw (pd.DataFrame) \
        \ndates (pd.Index) \
        \nprices (pd.DataFrame) \
        \nreturnns (pd.DataFrame)'


print(info)