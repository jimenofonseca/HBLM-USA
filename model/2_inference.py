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
import sys

# sys.path.append(r'E:\GitHub\HBLM-USA')
# os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
# os.environ["MKL_THREADING_LAYER"] = "GNU"

import pickle
import time
from sklearn.preprocessing import StandardScaler
import pymc3 as pm
import pandas as pd
from configuration import DATA_IPCC_REPORT_HBLM_TRAINING_FILE, DATA_IPCC_REPORT_HBLM_MODEL_FILE
from pointers import TRAINING_DATA_FILE_PATH, METADATA_FILE_PATH, BUILDING_PERFORMANCE_FOLDER_PATH, MODEL_FILE_PATH
from model.constants import response_variable, predictor_variables, samples

def main():

    # local variables
    output_model_path = MODEL_FILE_PATH
    data_training_df = pd.read_csv(TRAINING_DATA_FILE_PATH)

    # scale the data
    scaler = StandardScaler()
    fields_to_scale = [response_variable] + predictor_variables
    data_training_df[fields_to_scale] = pd.DataFrame(scaler.fit_transform(data_training_df[fields_to_scale]),
                                                columns=data_training_df[fields_to_scale].columns)


    # CREATE INDEXES FOR THE HIERACHY
    degree_index = data_training_df.groupby('CLIMATE_ZONE').all().reset_index().reset_index()[['index', 'CLIMATE_ZONE']]
    degree_state_index = data_training_df.groupby(["CLIMATE_ZONE", "CITY"]).all().reset_index().reset_index()[['index', "CLIMATE_ZONE", "CITY"]]
    degree_state_county_index = data_training_df.groupby(["CLIMATE_ZONE", "CITY", "BUILDING_CLASS"]).all().reset_index().reset_index()[['index', "CLIMATE_ZONE", "CITY", "BUILDING_CLASS"]]

    degree_state_indexes_df = pd.merge(degree_index, degree_state_index, how='inner', on='CLIMATE_ZONE', suffixes=('_d', '_ds'))
    degree_state_county_indexes_df = pd.merge(degree_state_indexes_df, degree_state_county_index, how='inner', on=['CLIMATE_ZONE', 'CITY'])
    indexed_salary_df = pd.merge(data_training_df, degree_state_county_indexes_df, how='inner', on=['CLIMATE_ZONE', 'CITY', 'BUILDING_CLASS']).reset_index()

    degree_indexes = degree_index['index'].values
    degree_count = len(degree_indexes)
    degree_state_indexes = degree_state_indexes_df['index_d'].values
    degree_state_count = len(degree_state_indexes)
    degree_state_county_indexes = degree_state_county_indexes_df['index_ds'].values
    degree_state_county_count = len(degree_state_county_indexes)

    # data_training_df[response_variable] = data_training_df[response_variable].astype(theano.config.floatX)

    with pm.Model() as hierarchical_model:

        # log(y) = alfa + beta*x1+ gamma*x2 + eps
        country_beta_mean = pm.Normal('country_beta_mean', mu=0, sd=100**2)
        country_beta_sd = pm.HalfCauchy('country_beta_sd', 5)
        country_alfa_mean = pm.Normal('country_alfa_mean', mu=0, sd=100**2)
        country_alfa_sd = pm.HalfCauchy('country_alfa_sd', 5)
        country_gamma_mean = pm.Normal('country_gamma_mean', mu=0, sd=100**2)
        country_gamma_sd = pm.HalfCauchy('country_gamma_sd', 5)

        climate_zone_beta_mean = pm.Normal("climate_zone_beta_mean", mu=country_beta_mean, sd=country_beta_sd, shape=degree_count)
        climate_zone_beta_sd = pm.HalfCauchy('climate_zone_beta_sd', 5, shape=degree_count)

        climate_zone_alfa_mean = pm.Normal("climate_zone_alfa_mean", mu=country_alfa_mean, sd=country_alfa_sd, shape=degree_count)
        climate_zone_alfa_sd = pm.HalfCauchy('climate_zone_alfa_sd', 5, shape=degree_count)

        climate_zone_gamma_mean = pm.Normal("climate_zone_gamma_mean", mu=country_gamma_mean, sd=country_gamma_sd, shape=degree_count)
        climate_zone_gamma_sd = pm.HalfCauchy('climate_zone_gamma_sd', 5, shape=degree_count)

        city_beta_mean = pm.Normal('city_beta_mean', mu=climate_zone_beta_mean[degree_state_indexes], sd=climate_zone_beta_sd[degree_state_indexes], shape=degree_state_count)
        city_beta_sd = pm.HalfCauchy('city_beta_sd', 5, shape=degree_state_count)

        city_alfa_mean = pm.Normal('city_alfa_mean', mu=climate_zone_alfa_mean[degree_state_indexes], sd=climate_zone_alfa_sd[degree_state_indexes], shape=degree_state_count)
        city_alfa_sd = pm.HalfCauchy('city_alfa_sd', 5, shape=degree_state_count)

        city_gamma_mean = pm.Normal('city_gamma_mean', mu=climate_zone_gamma_mean[degree_state_indexes], sd=climate_zone_gamma_sd[degree_state_indexes], shape=degree_state_count)
        city_gamma_sd = pm.HalfCauchy('city_gamma_sd', 5, shape=degree_state_count)

        building_sector_beta_mean = pm.Normal('building_sector_beta_mean',  mu=city_beta_mean[degree_state_county_indexes], sd=city_beta_sd[degree_state_county_indexes], shape=degree_state_county_count)
        building_sector_alfa_mean = pm.Normal('building_sector_alfa_mean', mu=city_alfa_mean[degree_state_county_indexes], sd=city_alfa_sd[degree_state_county_indexes], shape=degree_state_county_count)
        building_sector_gamma_mean = pm.Normal('building_sector_gamma_mean', mu=city_gamma_mean[degree_state_county_indexes], sd=city_gamma_sd[degree_state_county_indexes], shape=degree_state_county_count)

        eps = pm.HalfCauchy('eps', 5)
        y_obs = indexed_salary_df[response_variable].values
        x1 = indexed_salary_df[predictor_variables[0]].values
        x2 = indexed_salary_df[predictor_variables[1]].values

        building_mean = building_sector_alfa_mean[indexed_salary_df['index'].values] + building_sector_beta_mean[indexed_salary_df['index'].values] * x1 + building_sector_gamma_mean[indexed_salary_df['index'].values] * x2

        # Data likelihood
        y_like = pm.Normal('y_like', mu=building_mean, sd=eps, observed=y_obs)

    with hierarchical_model:
        #hierarchical_trace = pm.fit(50000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        hierarchical_trace = pm.sample(draws=samples, tune=1000, cores=2, nuts_kwargs=dict(target_accept=0.97))
        #
        #
        # save to disc
        with open(output_model_path, 'wb') as buff:
            pickle.dump({'inference': hierarchical_model, 'trace': hierarchical_trace,
                         'scaler': scaler, 'city_index_df': degree_state_county_indexes_df,
                         'response_variable': response_variable, 'predictor_variables': predictor_variables}, buff)


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = round((time.time() - t0)/60,2)
    print("finished after {} minutes".format(t1))
