'''
MIT License

Copyright (c) 2020 Jimeno A. Fonseca

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os

import pandas as pd
from sklearn import mixture

from model.constants import hours_of_day, air_density_kgm3, storey_height_m, \
    random_state, n_clusters, ZONE_NAMES, ACH_Residential, ACH_Commercial
from pointers import WEATHER_DATA_FOLDER_PATH

def calc_weight_climate_zone(sector, climate_region, floor_area_climate_df):
    floor_area_climate = floor_area_climate_df.loc[climate_region]
    return floor_area_climate['GFA_mean_' + sector + '_perc']

def calc_ACH_category(building_class):
    if building_class == "Residential":
        ACH_1_h = ACH_Residential
    elif building_class == "Commercial":
        ACH_1_h = ACH_Commercial
    return ACH_1_h


def calc_group_of_climate_zones(climate):
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)
    return new_clima


def read_weather_data_scenario(city, scenario):
    # Quantities
    weather_file_name = city.split(",")[0] + "_" + city.split(", ")[-1] + "-hour.dat"
    weather_file_name = weather_file_name.replace(" ", "_")
    weather_file_location = os.path.join(WEATHER_DATA_FOLDER_PATH, scenario, weather_file_name)
    weather_file = pd.read_csv(weather_file_location, sep='\s+', header=2, skiprows=0)
    temperatures_out_C = weather_file["Ta"].values[:8760]
    relative_humidity_percent = weather_file["RH"].values[:8760]

    return temperatures_out_C, relative_humidity_percent


def calc_clusters(data):
    building_classes = data.BUILDING_CLASS.unique()
    df = pd.DataFrame()
    for j, building_class in enumerate(building_classes):
        df3 = data[data["BUILDING_CLASS"] == building_class]
        if df3.empty == False:
            X_cluster = df3[["LOG_SITE_EUI_kWh_m2yr"]].values
            cv_type = 'tied'
            gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type=cv_type, random_state=random_state)
            gmm.fit(X_cluster)
            means = gmm.means_.T[0]  # /gmm.means_.T[1]
            cluster_labels = gmm.predict(X_cluster)
            df3['CLUSTER_LOG_SITE_EUI_kWh_m2yr'] = [round(means[cluster], 2) for cluster in cluster_labels]
            df = pd.concat([df, df3], ignore_index=True)

    df = data.merge(df[['BUILDING_ID', 'CLUSTER_LOG_SITE_EUI_kWh_m2yr']], on='BUILDING_ID')

    return df


def calc_thermal_consumption(DEG_kJperKgperday, COP, ACH, GFA):
    specific_thermal_consumption_kWhyr = (storey_height_m * air_density_kgm3 * ACH * GFA *
                                          DEG_kJperKgperday * hours_of_day) / (COP * 3600)

    return specific_thermal_consumption_kWhyr
