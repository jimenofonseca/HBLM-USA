import math
import numpy as np
import pandas as pd

from configuration import CONFIG_FILE, \
    DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, \
    DATA_IPCC_REPORT_HBLM_PREFINAL_FILE, DATA_IPCC_REPORT_HBLM_FINAL_FILE


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def main(scenarios, output_path):
    # group per scenario and building class and calculate mean and variance
    data_consumption = pd.read_csv(DATA_IPCC_REPORT_HBLM_PREFINAL_FILE)
    data_consumption = data_consumption.groupby(["BUILDING_CLASS", "SCENARIO"], as_index=False).agg(
        [percentile(50), percentile(2.5), percentile(97.5)])

    final_df = pd.DataFrame()
    for scenario in scenarios:
        ipcc_scenario_name = parse_scenario_name(scenario)
        year_scenario = scenario.split("_")[-1]
        for sector in ['Residential', 'Commercial']:
            mean_EJ = data_consumption.loc[sector, scenario]['TOTAL_ENERGY_EJ', 'percentile_50']
            est_97_5_EJ = data_consumption.loc[sector, scenario]['TOTAL_ENERGY_EJ', 'percentile_97.5']
            est_2_5_EJ = data_consumption.loc[sector, scenario]['TOTAL_ENERGY_EJ', 'percentile_2.5']

            dict_mean = {'Model': 'HBLM-USA 1.0',
                         'Region': 'USA',
                         'Unit': 'EJ_yr',
                         'Variable': 'Final Energy|Buildings|' + sector +'|Space',
                         'Scenario': ipcc_scenario_name + ' - 50th percentile',
                         'Year': year_scenario,
                         'Value': mean_EJ,
                         }
            dict_min = {'Model': 'HBLM-USA 1.0',
                        'Region': 'USA',
                        'Unit': 'EJ_yr',
                        'Variable': 'Final Energy|Buildings|' + sector + '|Space',
                        'Scenario': ipcc_scenario_name + ' - 2.5th percentile',
                        'Year': year_scenario,
                        'Value': est_2_5_EJ,
                        }
            dict_max = {'Model': 'HBLM-USA 1.0',
                        'Region': 'USA',
                        'Unit': 'EJ_yr',
                        'Variable': 'Final Energy|Buildings|' + sector + '|Space',
                        'Scenario': ipcc_scenario_name + ' - 97.5th percentile',
                        'Year': year_scenario,
                        'Value': est_97_5_EJ,
                        }
            dataframe = pd.DataFrame([dict_mean, dict_min, dict_max])
            final_df = pd.concat([final_df, dataframe], ignore_index=True)
    result = pd.pivot_table(final_df, values='Value', columns='Year', index=['Model', 'Scenario','Region', 'Variable', 'Unit'])
    result.to_csv(output_path)


def parse_scenario_name(scenario):
    mapa = {'A1B': 'Medium Impact',
            'A2': 'High Impact',
            'B1': 'Low Impact'
            }
    name = scenario.split('_')[-2]
    return mapa[name]


def calc_mean_two_gaussians(mean_1, mean_2):
    mean = mean_1 * mean_2
    return mean


def calc_std_two_gaussians(mean_both, mean_1, mean_2, std_1, std_2):
    Rstd = math.sqrt((std_1 / mean_1) ** 2 + (std_2 / mean_2) ** 2)
    std = Rstd * mean_both
    return std


if __name__ == "__main__":
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
    output_path = DATA_IPCC_REPORT_HBLM_FINAL_FILE
    cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_per_state')['City'].values
    main(scenarios, output_path)
