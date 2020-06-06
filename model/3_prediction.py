import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pandas as pd
import pymc3 as pm
from configuration import CONFIG_FILE, DATA_IPCC_REPORT_HBLM_PREDICTION_FILE, \
    DATA_FUTURE_EFFICIENCY_FILE, DATA_IPCC_REPORT_HBLM_MODEL_FILE, DATA_IPCC_REPORT_HBLM_PREFINAL_FILE


def input_data(Xy_prediction_path, response_variable, fields_to_scale, scaler):
    # READ DATA
    X_prediction = pd.read_csv(Xy_prediction_path)
    X_prediction[response_variable] = 0.0  # create dummy of response variable so we can scale all the data

    if scaler != None:
        X_prediction[fields_to_scale] = pd.DataFrame(scaler.transform(X_prediction[fields_to_scale]),
                                                     columns=X_prediction[fields_to_scale].columns)
    X_prediction.drop(response_variable, inplace=True, axis=1)
    return X_prediction


def do_prediction(Xy_predict, alpha, beta, gamma, response_variable, predictor_variables,
                  fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_predict[predictor_variables[0]].values
    x2 = Xy_predict[predictor_variables[1]].values
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


def main(input_model_path, prediction_data, main_cities, output_path):
    # get model data
    data_coefficients, \
    degree_index, \
    fields_to_scale, \
    n_samples, \
    predictor_variables, \
    response_variable, \
    scaler = get_model_data(input_model_path)

    # get floor areas
    data_floor_area = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="floor area").set_index('year')

    # get prediction data
    prediction_data_df = input_data(prediction_data, response_variable, fields_to_scale, scaler)

    # get cities
    main_cities = prediction_data_df['CITY'].unique()
    scenarios = prediction_data_df['SCENARIO'].unique()
    data_floor_area = pd.read_excel(DATA_FUTURE_EFFICIENCY_FILE, sheet_name="floor area").set_index('year')
    x_prediction_final = pd.DataFrame()
    alphas = []
    betas = []
    gammas = []
    areas = []
    for city in main_cities:
        X_prediction = prediction_data_df[prediction_data_df['CITY'] == city]
        for sector in ["Residential", "Commercial"]:
            X_predict_sector2 = X_prediction[X_prediction["BUILDING_CLASS"] == sector]
            if X_predict_sector2.empty or X_predict_sector2.empty:
                print(city, sector, "does not exist, we are skipping it")
            else:
                index_sector = \
                degree_index[(degree_index["CITY"] == city) & (degree_index["BUILDING_CLASS"] == sector)].index.values[
                    0]

                for scenario in scenarios:
                    # areas of USA
                    n = 100
                    year_scenario = scenario.split("_")[-1]
                    data_floor_area_scenario = data_floor_area.loc[float(year_scenario)]
                    # areas
                    mean_m2 = data_floor_area_scenario['GFA_mean_' + sector + '_m2']
                    std_m2 = data_floor_area_scenario['GFA_sd_' + sector + '_m2']
                    GFA_m2 = np.tile(np.random.normal(mean_m2, std_m2, n), n_samples)


                    #predictions
                    X_predict_sector = X_predict_sector2[X_predict_sector2['SCENARIO'] == scenario]

                    # get the median building in terms of GFA
                    X_predict_sector.sort_values(by='GROSS_FLOOR_AREA_m2', inplace=True)
                    below_median = pd.DataFrame(X_predict_sector[
                                                    X_predict_sector['GROSS_FLOOR_AREA_m2'] > X_predict_sector[
                                                        'GROSS_FLOOR_AREA_m2'].median()].iloc[0]).T
                    x_prediction_final = x_prediction_final.append([below_median] * (n_samples*n), ignore_index=True)

                    # append to dataframe final
                    alphas.extend(np.tile(data_coefficients['building_sector_alfa_mean__' + str(index_sector)].values, n))
                    betas.extend(np.tile(data_coefficients['building_sector_beta_mean__' + str(index_sector)].values, n))
                    gammas.extend(np.tile(data_coefficients['building_sector_gamma_mean__' + str(index_sector)].values, n))

                    # append areas
                    areas.extend(GFA_m2)
        print("city done: ", city)

    # do predictions
    response_variable_final = response_variable.split("_", 1)[1]  # take out the log name
    x_prediction_final[response_variable_final] = do_prediction(x_prediction_final,
                                                                alphas,
                                                                betas,
                                                                gammas,
                                                                response_variable,
                                                                predictor_variables,
                                                                fields_to_scale,
                                                                scaler)

    # save results per city and scenario
    x_prediction_final['EUI_kWh_m2yr'] = x_prediction_final[response_variable_final] / x_prediction_final['GROSS_FLOOR_AREA_m2']
    x_prediction_final['TOTAL_ENERGY_EJ'] = x_prediction_final['EUI_kWh_m2yr'] * areas * 3.6E-12
    x_prediction_final[["BUILDING_CLASS", "SCENARIO", "CITY", 'TOTAL_ENERGY_EJ']].to_csv(output_path)


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
    input_model_path = DATA_IPCC_REPORT_HBLM_MODEL_FILE
    output_path = DATA_IPCC_REPORT_HBLM_PREFINAL_FILE
    prediction_data = DATA_IPCC_REPORT_HBLM_PREDICTION_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    main(input_model_path, prediction_data, main_cities, output_path)
