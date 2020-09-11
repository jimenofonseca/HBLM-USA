# HBLM-USA-1.0
![GitHub license](https://img.shields.io/github/license/JIMENOFONSECA/HBLM-USA) ![Repo Size](https://img.shields.io/github/repo-size/JIMENOFONSECA/HBLM-USA) ![Lines](https://raw.githubusercontent.com/JIMENOFONSECA/HBLM-USA/image-data/badge-lines-of-code.svg)

Data-driven forecasting model of building energy consumption for the USA (part of the 6th IPCC report on Climate Change (2021)).

## How does it work

The model consists of a Hierachical Bayesian Linear Model trained over half a million buildings across the United States.
The forecasting model is based on the publication of [`Fonseca et al., 2020`](https://doi.org/10.3929/ethz-b-000416084) 
which includes forecasts for over 100 cities across the United States and for multiple scenarios of climate change.

![summary](https://github.com/JIMENOFONSECA/HBLM-USA/blob/master/images/summary.PNG)

This repository includes some post-processing steps needed to aggregate the original results of `Fonseca et al., 2020` 
at the national level (as requested by the IPCC database). These included:

1. Estimation of a weighted average of specific thermal energy consumption across different climatic regions. The weight is the built area per region. This estimation is carried out for every scenario of climate change described in `Fonseca et al., 2020`.
2. Montecarlo simulation of the built area for commercial and residential areas in the united states. The montecarlo simulation is carried out based on mean and standard errors of built area provided by the U.S. governmnet.
3. Multiplication of the results of 1 and 2 and estimation of the 50th, 2.5th and 97.5th percentiles for every climate change scenario.

## Installation

- Clone this repository
- Install dependencies

  - [`EnthalpyGradients==1.0`](https://pypi.org/project/EnthalpyGradients/)
  - `numpy`
  - `pandas`
  - `Scikit-learn==0.20.0`
  - `PyMC3==3.6`
  
- extract the file "/data/dummy_data_extract_here.zip"

## FAQ

- Where are the results stored? A: the results are inside the results folder / final_results.csv
- Where is the orginal database of built areas? A: It is publicly available for commercial and residential buildings [`here`](https://www.eia.gov/consumption/commercial/data/2012/) and [`here`](https://www.eia.gov/consumption/residential/data/2015/).
- Where is the weather data? A: The data is available from a private vendor [`Meteonorm`](https://meteonorm.com/en/). The data needs to be purchased separately for each one of the weather stations described in the file /inputs/metadata.xls. For convenience, I included a sample of the file when you extract "/data/dummy_data_extract_here.zip"
- Where is the building performance data? A: It is publicly available for close to 1 million buildings at [`LBL`](https://buildings.lbl.gov/cbs/bpd). The data needs to be downloaded via the API fo the LBL, you will need to ask LBL for access to the data. The data needs to be acquired for each one of the cities described in the file /inputs/metadata.xls. For convenience, I included a sample of the file when you extract "/data/dummy_data_extract_here.zip"

## Cite

J. A. Fonseca, I. Nevat and G.W. Peters, Quantifying Uncertainty in the Impact of Climate Change on Building Energy 
Consumption Across the United States, Appl. Energy, (2020).https://doi.org/10.3929/ethz-b-0004160848
