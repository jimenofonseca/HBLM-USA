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
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import time
import numpy as np
import pandas as pd
import pymc3 as pm
from pointers import METADATA_FILE_PATH, PREDICTION_DATA_FILE_PATH, MODEL_FILE_PATH, INTERMEDIATE_RESULT_FILE_PATH


def input_data(Xy_prediction_path, response_variable, fields_to_scale, scaler):
    # READ DATA
    X_prediction = pd.read_csv(Xy_prediction_path)
    X_prediction[response_variable] = 0.0  # create dummy of response variable so we can scale all the data

    if scaler != None:
        X_prediction[fields_to_scale] = pd.DataFrame(scaler.transform(X_prediction[fields_to_scale]),
                                                     columns=X_prediction[fields_to_scale].columns)
    X_prediction.drop(response_variable, inplace=True, axis=1)
    return X_prediction


def do_prediction(Xy_predict, response_variable, predictor_variables,
                  fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_predict[predictor_variables[0]].values
    x2 = Xy_predict[predictor_variables[1]].values
    alpha = Xy_predict["ALPHA"].values
    beta = Xy_predict["BETA"].values
    gamma = Xy_predict["GAMMA"].values
    Xy_predict[response_variable] = alpha + beta * x1 + gamma * x2

    # scale back
    if scaler != None:
        Xy_predict = pd.DataFrame(scaler.inverse_transform(Xy_predict[fields_to_scale]),
                                  columns=Xy_predict[fields_to_scale].columns)

    # scale back from log if necessry
    if response_variable.split("_")[0] == "LOG":
        y_prediction = np.exp(Xy_predict[response_variable].values.astype(float))
    else:
        y_prediction = Xy_predict[response_variable]

    return y_prediction


def calc_a_b_g_building(building_class, city, degree_index, data_coefficients):
    index_sector = \
        degree_index[(degree_index["CITY"] == city) & (degree_index["BUILDING_CLASS"] == building_class)].index.values[
            0]
    a = data_coefficients['building_sector_alfa_mean__' + str(index_sector)].values
    b = data_coefficients['building_sector_beta_mean__' + str(index_sector)].values
    g = data_coefficients['building_sector_gamma_mean__' + str(index_sector)].values

    return a, b, g


def main():
    # local variables
    input_model_path = MODEL_FILE_PATH
    prediction_data_path = PREDICTION_DATA_FILE_PATH
    output_path = INTERMEDIATE_RESULT_FILE_PATH

    scenarios_array = pd.read_excel(METADATA_FILE_PATH, sheet_name='SCENARIOS')['SCENARIO'].values
    cities_array = pd.read_excel(METADATA_FILE_PATH, sheet_name='CITIES')['CITY'].values

    floor_area_predictions_df = pd.read_excel(METADATA_FILE_PATH, sheet_name="FLOOR_AREA").set_index('year')

    # get model data
    data_coefficients, \
    indexes, \
    fields_to_scale, \
    n_samples, \
    predictor_variables, \
    response_variable, \
    scaler = get_model_data(input_model_path)

    # get prediction data
    prediction_data_df = input_data(prediction_data_path, response_variable, fields_to_scale, scaler)

    # get coefficients
    prediction_data_df = assign_coefficients(cities_array, data_coefficients, indexes, n_samples,
                                             prediction_data_df)

    # do predictions
    response_variable_final = response_variable.split("_", 1)[1]  # take out the log name
    prediction_data_df[response_variable_final] = do_prediction(prediction_data_df,
                                                                response_variable,
                                                                predictor_variables,
                                                                fields_to_scale,
                                                                scaler)
    prediction_data_df['EUI_kWh_m2yr'] = prediction_data_df[response_variable_final] / prediction_data_df['GROSS_FLOOR_AREA_m2']

    #calcuilate weighted average
    data_weighted_average_df = calc_weighted_average_per_scenario(prediction_data_df)

    # calculate the energy consumption per scenario incorporating variance in built areas
    data_final_df = calc_total_energy_consumption_per_scenario(data_weighted_average_df,
                                                               floor_area_predictions_df,
                                                               scenarios_array)

    #save the results to disk
    data_final_df.to_csv(output_path, index=False)
    print("done")


def calc_total_energy_consumption_per_scenario(data_weighted_average_df, floor_area_predictions_df, scenarios_array):
    data_final_df = pd.DataFrame()
    for scenario in scenarios_array:
        data_scenario = data_weighted_average_df[data_weighted_average_df["SCENARIO"] == scenario]
        year = data_scenario['YEAR'].values[0]
        data_floor_area_scenario = floor_area_predictions_df.loc[float(year)]
        for sector in ['Residential', 'Commercial']:
            # calculate totals
            eui_kWhm2yr = data_scenario[data_scenario["BUILDING_CLASS"] == sector]['EUI_kWh_m2yr'].values[0]

            # add uncertainty in total built area
            mean_m2 = data_floor_area_scenario['GFA_mean_' + sector + '_m2']
            std_m2 = data_floor_area_scenario['GFA_sd_' + sector + '_m2']
            GFA_m2 = np.random.normal(mean_m2, std_m2, 100)

            total_consumption_EJ = GFA_m2 * eui_kWhm2yr * 3.6E-12

            # list of fields to extract
            dict_data = pd.DataFrame({"SCENARIO": scenario,
                                      "YEAR": year,
                                      "GFA_Bm2": GFA_m2/1E9,
                                      "BUILDING_CLASS": sector,
                                      "EUI_kWh_m2_yr": eui_kWhm2yr,
                                      "TOTAL_CONSUMPTION_EJ": total_consumption_EJ})
            data_final_df = pd.concat([data_final_df, dict_data], ignore_index=True)
    return data_final_df


def calc_weighted_average_per_scenario(specific_thermal_consumption_per_city_df):
    data_mean_per_scenario = specific_thermal_consumption_per_city_df.groupby(
        ["YEAR", "BUILDING_CLASS", "SCENARIO", "CLIMATE_REGION"],
        as_index=False).agg('mean')
    data_mean_per_scenario["EUI_kWh_m2yr"] = data_mean_per_scenario["EUI_kWh_m2yr"] * data_mean_per_scenario["WEIGHT"]
    data_weighted_average = data_mean_per_scenario.groupby(["YEAR", "BUILDING_CLASS", "SCENARIO"], as_index=False).agg(
        'sum')
    return data_weighted_average


def assign_coefficients(cities_array, data_coefficients, indexes, n_samples, prediction_data_df):
    x_prediction_final = pd.DataFrame()
    alphas = []
    betas = []
    gammas = []
    for city in cities_array:
        X_prediction = prediction_data_df[prediction_data_df['CITY'] == city]
        for sector in ["Residential", "Commercial"]:
            X_predict_sector = X_prediction[X_prediction["BUILDING_CLASS"] == sector]
            if X_predict_sector.empty or X_predict_sector.empty:
                print(city, sector, "does not exist, we are skipping it")
            else:
                index_sector = indexes[(indexes["CITY"] == city) &
                                       (indexes["BUILDING_CLASS"] == sector)].index.values[0]
                x_prediction_final = x_prediction_final.append([X_predict_sector] * n_samples, ignore_index=True)

                # append to dataframe final
                n = X_predict_sector.shape[0]
                alphas.extend(np.tile(data_coefficients['building_sector_alfa_mean__' + str(index_sector)].values,n))
                betas.extend(np.tile(data_coefficients['building_sector_beta_mean__' + str(index_sector)].values,n))
                gammas.extend(np.tile(data_coefficients['building_sector_gamma_mean__' + str(index_sector)].values,n))

        print("coefficients assigned to city", city)
    x_prediction_final['ALPHA'] = alphas
    x_prediction_final['BETA'] = betas
    x_prediction_final['GAMMA'] = gammas
    return x_prediction_final


def get_model_data(input_model_path):
    # loading model
    with open(input_model_path, 'rb') as buff:
        data_coefficients = pickle.load(buff)
        hierarchical_model, \
        hierarchical_trace, \
        scaler, \
        degree_index, \
        response_variable, predictor_variables = data_coefficients['inference'], \
                                                 data_coefficients['trace'], \
                                                 data_coefficients['scaler'], \
                                                 data_coefficients['city_index_df'], \
                                                 data_coefficients['response_variable'], \
                                                 data_coefficients['predictor_variables']
    # fields to scale, get data_coefficients of traces
    fields_to_scale = [response_variable] + predictor_variables
    # get variables
    data_coefficients = pm.trace_to_dataframe(hierarchical_trace)
    data_coefficients = data_coefficients.sample(n=100).reset_index(drop=True)
    n_samples = data_coefficients.shape[0]
    return data_coefficients, degree_index, fields_to_scale, n_samples, predictor_variables, response_variable, scaler


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = round((time.time() - t0) / 60, 2)
    print("finished after {} minutes".format(t1))
