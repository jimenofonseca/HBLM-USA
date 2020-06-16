'''
MIT License

Copyright (c) 2020 Jimeno Fonseca

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

import numpy as np
import pandas as pd
import time
from enthalpygradients import EnthalpyGradient

from model.auxiliary import read_weather_data_scenario, calc_thermal_consumption, calc_clusters, calc_ACH_category, \
    calc_group_of_climate_zones
from model.constants import COP_heating, COP_cooling, T_base_cooling_C, T_base_heating_C, RH_base_cooling_perc, \
    RH_base_heating_perc
from pointers import TRAINING_DATA_FILE_PATH, METADATA_FILE_PATH, BUILDING_PERFORMANCE_FOLDER_PATH


def main():
    # local variables
    data_energy_folder_path = BUILDING_PERFORMANCE_FOLDER_PATH
    output_path = TRAINING_DATA_FILE_PATH
    cities_array = pd.read_excel(METADATA_FILE_PATH, sheet_name='CITIES')['CITY'].values

    # get climate climate_zone classification (grouped version) - as in the original paper
    climate_regions_array = pd.read_excel(METADATA_FILE_PATH, sheet_name='CITIES')['Climate Region'].values
    climate_zones_array = pd.read_excel(METADATA_FILE_PATH, sheet_name='CITIES')['CLIMATE'].values
    climate_zones_array = calc_group_of_climate_zones(climate_zones_array)

    final_df = pd.DataFrame()
    for city, climate_zone, climate_region in zip(cities_array, climate_zones_array, climate_regions_array):
        # read wheater data

        T_outdoor_C, RH_outdoor_perc = read_weather_data_scenario(city, "data_1990_2010")

        # calculate specific energy consumption with daily enthalpy gradients model
        eg = EnthalpyGradient(T_base_cooling_C, RH_base_cooling_perc)
        DEG1 = eg.enthalpy_gradient(T_outdoor_C, RH_outdoor_perc, type='cooling')
        DEG2 = eg.enthalpy_gradient(T_outdoor_C, RH_outdoor_perc, type='dehumidification')

        eg = EnthalpyGradient(T_base_heating_C, RH_base_heating_perc)
        DEG3 = eg.enthalpy_gradient(T_outdoor_C, RH_outdoor_perc, type='heating')
        DEG4 = eg.enthalpy_gradient(T_outdoor_C, RH_outdoor_perc, type='humidification')

        DEG = DEG1 + DEG2 + DEG3 + DEG4

        # get measured data
        # get_local_data:
        data_measured = pd.read_csv(os.path.join(data_energy_folder_path, city + ".csv"))
        data_measured["BUILDING_ID"] = [city + str(ix) for ix in data_measured.index]
        data_measured['SCENARIO'] = "data_1990_2010"
        data_measured['CLIMATE_REGION'] = climate_region
        data_measured['CLIMATE_ZONE'] = climate_zone
        data_measured["GROSS_FLOOR_AREA_m2"] = (data_measured["floor_area"] * 0.092903).values
        data_measured["BUILDING_CLASS"] = data_measured["building_class"].values
        data_measured["CITY"] = city
        data_measured["ACH"] = data_measured['BUILDING_CLASS'].apply(lambda x: calc_ACH_category(x))
        data_measured["SITE_ENERGY_kWh_yr"] = (data_measured["site_energy"] * 0.293071).round(2)
        data_measured["SITE_EUI_kWh_m2yr"] = data_measured["SITE_ENERGY_kWh_yr"] / data_measured["GROSS_FLOOR_AREA_m2"]
        data_measured["THERMAL_ENERGY_kWh_yr"] = data_measured.apply(lambda x:
                                                                     calc_thermal_consumption(DEG,
                                                                                              (COP_heating + COP_cooling) / 2,
                                                                                              x["ACH"],
                                                                                              x["GROSS_FLOOR_AREA_m2"]),
                                                                     axis=1)

        # log also
        data_measured["LOG_THERMAL_ENERGY_kWh_yr"] = np.log(data_measured["THERMAL_ENERGY_kWh_yr"].values)
        data_measured['LOG_SITE_EUI_kWh_m2yr'] = np.log(data_measured["SITE_EUI_kWh_m2yr"].values)
        data_measured['LOG_SITE_ENERGY_kWh_yr'] = np.log(data_measured["SITE_ENERGY_kWh_yr"].values)

        data = calc_clusters(data_measured)

        # list of fields to extract
        fields = ["BUILDING_ID",
                  "CITY",
                  "CLIMATE_REGION",
                  "CLIMATE_ZONE",
                  "SCENARIO",
                  "BUILDING_CLASS",
                  "GROSS_FLOOR_AREA_m2",
                  "LOG_SITE_EUI_kWh_m2yr",
                  "LOG_SITE_ENERGY_kWh_yr",
                  "LOG_THERMAL_ENERGY_kWh_yr",
                  "CLUSTER_LOG_SITE_EUI_kWh_m2yr"
                  ]
        final_df = pd.concat([final_df, data[fields]], ignore_index=True)
        print("city done: ", city)
    final_df.to_csv(output_path, index=False)
    print("done")
    return final_df


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = round((time.time() - t0)/60,2)
    print("finished after {} minutes".format(t1))
