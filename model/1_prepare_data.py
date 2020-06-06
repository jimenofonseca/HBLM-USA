import os

import pandas as pd

from IPCC_report.daily_enthalpy_gradient_module import daily_enthalpy_gradients
from IPCC_report.auxiliary import get_today_efficiency_conditions,\
    calc_clusters, calc_thermal_consumption, read_weather_data_scenario
from configuration import CONFIG_FILE, ZONE_NAMES,\
    DATA_FUTURE_EFFICIENCY_FILE,DATA_RAW_BUILDING_PERFORMANCE_FOLDER, \
    DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, \
    DATA_IPCC_REPORT_HBLM_PREDICTION_FILE, DATA_IPCC_REPORT_HBLM_TRAINING_FILE
import numpy as np


def calc_ACH_category(building_class):
    if building_class == "Residential":
        ACH_1_h = 4
    elif building_class == "Commercial":
        ACH_1_h = 6
    return ACH_1_h

def main_training(cities, climate, data_energy_folder, data_ipcc_folder, output_path):
    data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')
    scenario = "data_1990_2010"
    final_df = pd.DataFrame()
    # get climate calssification
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)

    for city, clim in zip(cities, new_clima):
        # read wheater data
        T_outdoor_C, RH_outdoor_perc = read_weather_data_scenario(city, data_ipcc_folder, scenario)

        # read boundary conditions
        COP_cooling, \
        COP_heating, \
        RH_base_cooling_perc, \
        RH_base_heating_perc, \
        T_base_cooling_C, \
        T_base_heating_C = get_today_efficiency_conditions(data_efficiency)

        # calculate enthalpy gradients
        DEG_C_kJperKgperday, \
        DEG_H_kJperKgperday, \
        DEG_DEHUM_kJperKgperday, \
        DEG_HUM_kJperKgperday = daily_enthalpy_gradients(T_outdoor_C,
                                                         RH_outdoor_perc,
                                                         T_base_heating_C,
                                                         T_base_cooling_C,
                                                         RH_base_heating_perc,
                                                         RH_base_cooling_perc)

        #calculate total
        DEG_kJperKgperday = DEG_C_kJperKgperday + DEG_H_kJperKgperday + DEG_DEHUM_kJperKgperday + DEG_HUM_kJperKgperday

        #get measured data
        # get_local_data:
        data_measured = pd.read_csv(os.path.join(data_energy_folder, city + ".csv"))
        data_measured["BUILDING_ID"] = [city + str(ix) for ix in data_measured.index]
        data_measured['SCENARIO'] = scenario
        data_measured['CLIMATE_ZONE'] = clim
        data_measured["GROSS_FLOOR_AREA_m2"] = (data_measured["floor_area"] * 0.092903).values
        data_measured["BUILDING_CLASS"] = data_measured["building_class"].values
        data_measured["CITY"] = city
        data_measured["ACH"] = data_measured['BUILDING_CLASS'].apply(lambda x: calc_ACH_category(x))
        data_measured["SITE_ENERGY_kWh_yr"] = (data_measured["site_energy"] * 0.293071).round(2)
        data_measured["SITE_EUI_kWh_m2yr"] = data_measured["SITE_ENERGY_kWh_yr"]/data_measured["GROSS_FLOOR_AREA_m2"]
        data_measured["THERMAL_ENERGY_kWh_yr"] = data_measured.apply(lambda x: calc_thermal_consumption(DEG_kJperKgperday,
                                                                                                       (COP_heating+COP_cooling)/2,
                                                                                                       x["ACH"],
                                                                                                       x["GROSS_FLOOR_AREA_m2"]), axis=1)

        #log also
        data_measured["LOG_THERMAL_ENERGY_kWh_yr"] = np.log(data_measured["THERMAL_ENERGY_kWh_yr"].values)
        data_measured['LOG_SITE_EUI_kWh_m2yr'] = np.log(data_measured["SITE_EUI_kWh_m2yr"].values)
        data_measured['LOG_SITE_ENERGY_kWh_yr'] = np.log(data_measured["SITE_ENERGY_kWh_yr"].values)

        data = calc_clusters(data_measured)

        # list of fields to extract
        fields = ["BUILDING_ID",
                  "CITY",
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
        print("scenario and city done: ", scenario, city)
    final_df.to_csv(output_path, index=False)
    print("done")
    return final_df

def main_prediction(cities, climate, data_energy_folder, data_ipcc_folder, output_path, scenarios, data_training):
    data_efficiency = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="data").set_index('year')
    final_df = pd.DataFrame()
    # get climate calssification
    new_clima = []
    for clima in climate:
        for category, categories in ZONE_NAMES.items():
            if clima.split(" ")[0] in categories:
                new_clima.append(category)

    for city, clim in zip(cities, new_clima):
        for scenario in scenarios:
            # read wheater data
            T_outdoor_C, RH_outdoor_perc = read_weather_data_scenario(city, data_ipcc_folder, scenario)

            # read boundary conditions
            COP_cooling, \
            COP_heating, \
            RH_base_cooling_perc, \
            RH_base_heating_perc, \
            T_base_cooling_C, \
            T_base_heating_C = get_today_efficiency_conditions(data_efficiency)

            # calculate enthalpy gradients
            DEG_C_kJperKgperday, \
            DEG_H_kJperKgperday, \
            DEG_DEHUM_kJperKgperday, \
            DEG_HUM_kJperKgperday = daily_enthalpy_gradients(T_outdoor_C,
                                                             RH_outdoor_perc,
                                                             T_base_heating_C,
                                                             T_base_cooling_C,
                                                             RH_base_heating_perc,
                                                             RH_base_cooling_perc)

            #calculate total
            DEG_kJperKgperday = DEG_C_kJperKgperday + DEG_H_kJperKgperday + DEG_DEHUM_kJperKgperday + DEG_HUM_kJperKgperday

            #get measured data
            # get_local_data:
            data_measured = pd.read_csv(os.path.join(data_energy_folder, city + ".csv"))
            data_measured["BUILDING_ID"] = [city + str(ix) for ix in data_measured.index]
            data_measured['SCENARIO'] = scenario
            data_measured['CLIMATE_ZONE'] = clim
            data_measured["GROSS_FLOOR_AREA_m2"] = (data_measured["floor_area"] * 0.092903).values
            data_measured["BUILDING_CLASS"] = data_measured["building_class"].values
            data_measured["CITY"] = city
            data_measured["ACH"] = data_measured['BUILDING_CLASS'].apply(lambda x: calc_ACH_category(x))
            data_measured["THERMAL_ENERGY_kWh_yr"] = data_measured.apply(lambda x: calc_thermal_consumption(DEG_kJperKgperday,
                                                                                                           (COP_heating+COP_cooling)/2,
                                                                                                           x["ACH"],
                                                                                                           x["GROSS_FLOOR_AREA_m2"]), axis=1)
            #log also
            data_measured["LOG_THERMAL_ENERGY_kWh_yr"] = np.log(data_measured["THERMAL_ENERGY_kWh_yr"].values)

            # list of fields to extract
            fields = ["BUILDING_ID",
                      "CITY",
                      "CLIMATE_ZONE",
                      "SCENARIO",
                      "BUILDING_CLASS",
                      "GROSS_FLOOR_AREA_m2",
                      "LOG_THERMAL_ENERGY_kWh_yr",
                      ]
            final_df = pd.concat([final_df, data_measured[fields]], ignore_index=True)
            print("scenario and city done: ", scenario, city)

    #now import the training dataset and get tremaining cluster_data
    data_training_df = pd.read_csv(data_training,usecols=["BUILDING_ID", "CLUSTER_LOG_SITE_EUI_kWh_m2yr"])
    data_final_df = final_df.merge(data_training_df, left_on='BUILDING_ID', right_on='BUILDING_ID')
    data_final_df.to_csv(output_path, index=False)
    print("done")
    return final_df


if __name__ == "__main__":

    # training dataset
    data_ipcc_folder = DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
    output_path = DATA_IPCC_REPORT_HBLM_TRAINING_FILE
    data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
    climate = pd.read_excel(CONFIG_FILE, sheet_name='cities_per_state')['climate'].values
    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_per_state')['City'].values
    main_training(cities, climate, data_energy_folder, data_ipcc_folder, output_path)

    # prediction dataset
    data_training = DATA_IPCC_REPORT_HBLM_TRAINING_FILE
    output_path = DATA_IPCC_REPORT_HBLM_PREDICTION_FILE
    scenarios = ["data_A1B_2010",
                 "data_A1B_2020",
                 "data_A1B_2030",
                 "data_A1B_2040",
                 "data_A1B_2050",
                 "data_A1B_2060",
                 "data_A1B_2070",
                 "data_A1B_2080",
                 "data_A1B_2090",
                 "data_A1B_2100",
                 "data_A2_2010",
                 "data_A2_2020",
                 "data_A2_2030",
                 "data_A2_2040",
                 "data_A2_2050",
                 "data_A2_2060",
                 "data_A2_2070",
                 "data_A2_2080",
                 "data_A2_2090",
                 "data_A2_2100",
                 "data_B1_2010",
                 "data_B1_2020",
                 "data_B1_2030",
                 "data_B1_2040",
                 "data_B1_2050",
                 "data_B1_2060",
                 "data_B1_2070",
                 "data_B1_2080",
                 "data_B1_2090",
                 "data_B1_2100"]
    main_prediction(cities, climate, data_energy_folder, data_ipcc_folder, output_path, scenarios, data_training)



