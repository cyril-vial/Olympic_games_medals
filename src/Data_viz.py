#region Import modules and cleaned dataset

import numpy as np
np.random.seed(1)
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
pd.set_option('display.max_columns', None) #Show all columns

path = r"\data\dataset_train_clean.csv"
df = pd.read_csv(path, encoding='latin-1', na_values=[])
df = df.fillna("None")

#endregion

#region Displaying a pie chart of the proportion of each unique value of 'Medal'

nb_medals = df["Medal"].value_counts().sum()
prop = (100 * df["Medal"].value_counts() / nb_medals).to_list()
names=["Gold","Bronze","Silver"]
plt.pie(prop, labels=names, labeldistance=1.15,autopct='%1.1f%%',wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })
plt.show()

#endregion

#region Plotting the number of medals for each OG in Summer and then in Winter

df["has_medal"] = df["Medal"].apply(lambda x: 0 if x=="None" else 1)


df_summer = df.loc[df["Season"]=="Summer"].reset_index()
summer_medal = df_summer.loc[df_summer["has_medal"]==1]
df_summer_medal = summer_medal.groupby("Year")["has_medal"].value_counts().to_frame().reset_index()
df_summer_medal = df_summer_medal.drop(columns="has_medal")
df_summer_medal.rename(columns={"count":"Number of medals"},inplace=True)
df_summer_medal = df_summer_medal.sort_values(by="Year")

ax0 = sns.barplot(data=df_summer_medal,x="Year",y="Number of medals")
plt.title("Number of medals per summer olympic game")
plt.figure(figsize = (12,5))
for i in ax0.containers:
    ax0.bar_label(i,rotation=80) # displaying values

df_winter = df.loc[df["Season"]=="Winter"].reset_index()
winter_medal = df_winter.loc[df_winter["has_medal"]==1]
df_winter_medal = winter_medal.groupby("Year")["has_medal"].value_counts().to_frame().reset_index()
df_winter_medal = df_winter_medal.drop(columns="has_medal")
df_winter_medal.rename(columns={"count":"Number of medals"},inplace=True)
df_winter_medal = df_winter_medal.sort_values(by="Year")

plt.figure(figsize = (12,5))
plt.title("Number of medals per winter olympic game")
ax1 = sns.barplot(data=df_winter_medal,x="Year",y="Number of medals")
for i in ax1.containers:
    ax1.bar_label(i,rotation=80) # displaying values

#endregion

#region Displaying the average number of medals won when a country hosts the summer Olympic Games after 1950 and when it does not

Summer=df[(df['Season']=='Summer') & (df['Year']>=1950)]
list_hosted=Summer[Summer['Won_host_country']==True]['Country'].unique()
host=Summer[Summer['Country'].isin(list_hosted)].groupby(['Country','Won_host_country','Year'])['has_medal'].sum().reset_index()
host_avg = int(host[host['Won_host_country']==True]["has_medal"].mean())
not_host_avg = int(host[host['Won_host_country']==False]["has_medal"].mean())

print("Mean when a country is host : ", host_avg)
print("Mean when a country is not host : ", not_host_avg)

# Now displaying this per country
df_host_plot=host.groupby(['Country','Won_host_country'])['has_medal'].mean().reset_index()
sns.barplot(x='Country', y='has_medal', hue='Won_host_country', data=df_host_plot)
plt.xticks(rotation=90)
plt.title('Average number of medals won by country')
plt.show()

# We notice, as expected, that the number of medals won is way higher when hosting the OG

#endregion

#region Computing correlation matrix between int and floats variables

corr_mat0 = df.select_dtypes(include=[int,float]).drop(columns=["has_medal"]).corr()

# Plotting the heatmap from the correlation matrix
plt.figure(figsize=(6,5))
plt.title("Correlation matrix in heatmap")

sns.heatmap(corr_mat0, annot=True, cmap="coolwarm")
plt.show()

#variables are not very correlated (abs<0.15) except for height and weight, which seems understandable

#endregion

#region Distribution of age among gold medal winners

gold = df[df["Medal"]=="Gold"]

plt.figure(figsize=(20, 10))
plt.tight_layout()
ax = sns.countplot(data=gold, x="Age")
plt.title('Distribution of Gold Medals')

# Iterating over the bars and display the count on top of each bar
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points')

plt.show()


fig, ax = plt.subplots(1,1, figsize=(10,4))
host_medal=df.groupby(['Year','Country','Won_host_country','has_medal'])['Medal'].count().reset_index()
sns.scatterplot(
    data=host_medal[host_medal["Won_host_country"]],
    x='Year',
    y='Medal',
    s=100,
    color='red',
    ax=ax
)

sns.scatterplot(
    data=host_medal[host_medal["Won_host_country"]==False],
    x='Year',
    y='Medal',
    s=75,
    color='black',
    alpha=0.2,
    ax=ax
)
Xstart, Xend = ax.get_xlim()
Ystart, Yend = ax.get_ylim()

ax.set_ylabel("Total Medals",loc='bottom',color='gray')
ax.set_xlabel("")

plt.show()

#endregion

#region Map of total medals per country

geo_df = df.groupby(['Country'])['has_medal'].sum().reset_index()
geo_df["has_medal_log10"] = np.log10(geo_df["has_medal"]+1)
geo_df

# Interactive choropleth map using plotly

import plotly.express as px
fig = px.choropleth(geo_df, locations="Country",
                    color="has_medal_log10", # enables more contrats between countries : use "has_medal" if you want only to see the number of medals
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma,
                    locationmode='country names')
fig.show()


# Now using folium

from folium.plugins import Search
from folium.plugins import HeatMap
import branca
import folium
from folium import plugins
import geopandas as gpd
geo_df = df.groupby(['Country', 'Medal']).size().unstack(fill_value=0)
geo_df['has_medal'] = geo_df['Gold'] + geo_df['Silver'] + geo_df['Bronze']
geo_df['has_medal'] = geo_df['has_medal'].fillna(0)
geo_df['has_medal_log10'] = np.log10(geo_df['has_medal']+1)
geo_df.reset_index(inplace=True)


# For geographic plotting
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'

global_polygons = gpd.read_file(country_shapes)
global_polygons.to_file('global_polygons.geojson', driver = 'GeoJSON')
global_polygons['name'] = global_polygons['name'].str.replace('United States of America', 'USA')
geo_df = pd.merge(global_polygons, geo_df, how='left', left_on='name', right_on='Country')
geo_df.drop(['name'], axis=1, inplace=True)
geo_df=geo_df.dropna()


def rd2(x):
    return round(x, 2)


minimum, maximum = geo_df["has_medal"].quantile([0.05, 0.95]).apply(rd2)
mean = round(geo_df["has_medal"].mean(), 2)

colormap = branca.colormap.LinearColormap(
    colors=["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"],
    index=geo_df["has_medal"].quantile([0.25, 0.5, 0.85, 0.95]),
    vmin=minimum,
    vmax=maximum,
)

colormap.caption = "Total medals"


m = folium.Map(location=(50,0),zoom_start=3)


def style_function(x):
    return {
        "fillColor": colormap(x["properties"]["has_medal"]),
        "color": "black",
        "weight": 2,
        "fillOpacity": 0.5,
    }

Map_Layer = folium.GeoJson(
    geo_df,
    name="Medals",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["Country", "has_medal", "Gold", "Silver", "Bronze"],  # Remove 'geometry' from fields
        aliases=["Country", "Total Medals", "Gold", "Silver", "Bronze"],
        localize=True
    ),
).add_to(m)



plugins.Search(Map_Layer,position='topleft',
                           search_zoom=5,placeholder="Search for a country",weight=3,
                           search_label='Country',
                           geom_type='Polygon').add_to(m)


minimap = plugins.MiniMap()
m.add_child(minimap)

folium.LayerControl().add_to(m)
colormap.add_to(m)

m

#endregion
