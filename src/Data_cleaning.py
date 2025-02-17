#region Import libraries

import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
pd.set_option('display.max_columns', None) #Show all columns

#endregion

#region Import dataset

path0 = r"\data\athlete_events.csv"
dataset = pd.read_csv(path0,encoding='latin-1')
# This dataset is a list of olympic games (summer & winter) performances from athletes and the medal they won (or not: NaN) plus a number of categorical and continuous variables (weight, age, country, etc.)

#endregion

#region Analyzing the dataframe's dtyopes

columns_int = dataset.select_dtypes("int64").columns.to_list()
columns_float = dataset.select_dtypes("float64").columns.to_list()
columns_object = dataset.select_dtypes("object").columns.to_list()

columns_int_object = columns_int + columns_object

dataset_unique=dataset[columns_int_object].nunique()
dataset_unique=dataset_unique.reset_index()
dataset_unique.columns=["Columns","Nb_unique"] # Computing the number of unique values for the columns with an object and int type.
dataset_unique.head()

dataset_unique.loc[dataset_unique["Nb_unique"] > 15] #displaying columns which have over 15 unique values

dataset.select_dtypes("float64").describe().round(2) #statistics on float type columns
# Age: Very low & High min&max value for Age
# Age & Weight: Very high max value (much higher than Q3/75%)

#endregion

#region Analyzing missing values

df_na = dataset.isna().sum().to_frame("Number_of_missing_values") # Computing the number of NaN values
df_na

df_na.loc[df_na["Number_of_missing_values"] > 0] # Columns having NaN values

dataset["Medal"] = dataset["Medal"].fillna("None") # Replacing "NaN" by "None" in the Medal column
dataset["Medal"]

# Filling "NaN" values of other column with its mode (this choice is actually important for the results of our model)
fill_age = dataset["Age"].mode().to_list()[0]
fill_weight = dataset["Weight"].mode().to_list()[0]
fill_height = dataset["Height"].mode().to_list()[0]
dataset["Age"] = dataset["Age"].fillna(fill_age)
dataset["Weight"] = dataset["Weight"].fillna(fill_weight)
dataset["Height"] = dataset["Height"].fillna(fill_height)

# Alternatively: Dropping columns having "NaN" values
dataset = dataset.dropna(axis=1).reset_index(drop=True)
dataset.head()


dataset.isna().sum() #checking that there are no more NaN values anymore


#region adding countries to the dataset
def preparation(dataset):
  regions = pd.read_csv(r"C:\Users\cyril\PycharmProjects\HiParisMLproject\data\noc_regions.csv")

  dict_regions = dict(zip(regions["NOC"].to_list(), regions["region"].to_list()))
  countries = dataset["NOC"].map(dict_regions)
  dataset.insert(8, "Country", countries)
  dict_host_country = {
    "Seoul":"South Korea",
    "Barcelona":"Spain",
    "Atlanta":"USA",
    "Sydney":"Australia",
    "Athina":"Greece",
    "Beijing":"China",
    "Melbourne":"Australia",
    "Los Angeles":'USA',
    "St. Louis": 'USA',
    "Moskva":"Russia",
    "Montreal":"Canada",
    "Munich" :"Germany",
    "Berlin":"Germany",
    "Mexico City":"Mexico",
    "Tokyo":"Japan",
    "Roma":"Italy",
    "Paris":"France",
    "Helsinki": "Finland",
    "Amsterdam":"Netherlands",
    "Antwerpen": "Belgium",
    "Stockholm": "Sweden",
    "Rio de Janeiro":"Bresil",
    "London":"UK",
    'Calgary':"Italy",
    'Albertville':"France",
    'Lillehammer':"Norway",
    'Salt Lake City':"USA",
    'Lake Placid':"USA",
    'Sochi':"Russia",
    'Nagano':"Japan",
    'Torino':"Italy",
    'Squaw Valley':"USA",
    'Innsbruck':"Austria",
    'Sarajevo':"Yugoslavia",
    'Oslo':"Sweden",
    "Cortina d'Ampezzo":"Italy",
    'Vancouver':"Canada",
    'Grenoble':"France",
    'Sapporo':"Japan",
    'Chamonix':"France",
    'Sankt Moritz':"Switzerland",
    'Garmisch-Partenkirchen':"Germany"}
  dataset["Host_country"] = dataset["City"].map(dict_host_country)
  dataset["Won_host_country"] = dataset["Host_country"] == dataset["Country"]
  dataset.reset_index(drop=True, inplace=True)
  return dataset

dataset = preparation(dataset)

#endregion

# Exporting our cleaned dataset
dataset.to_csv(r"C:\Users\cyril\PycharmProjects\HiParisMLproject\data\dataset_train_clean.csv", index=False, encoding="utf-8")


#endregion


