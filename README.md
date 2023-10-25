<h1 align="center">
  Predictive view of Bayesian Clustering
  <br/>
</h1>

<p align="center">Project for the course "Bayesian Statistics" at ENSAE Paris.  <br/> </p>

<p align="center">
    <a href="https://fr.overleaf.com/read/psbhmxhmpgst">
        <img src="https://img.shields.io/badge/Overleaf-47A141?style=for-the-badge&logo=Overleaf&logoColor=white" alt="SOURCE CODE"/> 
    </a>
</p>


## Overview

In this project, we implement two methods related to bayesian clustering described in the paper [A predictive view of Bayesian clustering]() by Fernando A. Quintana.

## Models

### Model-Based Clustering

## Files and folders

* The implementation is done in Python, and the code is available in the `./src/models` folder.
* The notebooks used to generate the results are available in the `notebooks` folder.
    - The notebook `MBC_viz.ipynb` contains results of the application of Model Based Clustering method
    - The notebook `DBC_viz.ipynb` contains results of Dirichlet based clustering method

## Data used

The data set `mcs_ds_edited_iter_shuffled.csv` used in the project is available in the `data` folder and is taken from the [UCI  Machine Learning Repo](https://archive.ics.uci.edu/ml/datasets/Average+Localization+Error+%28ALE%29+in+sensor+node+localization+process+in+WSNs#)

## References

The resources used for the project are the follows: 
```BibTeX
@article{quintana,
author = {Quintana, Fernando},
year = {2006},
month = {08},
pages = {2407-2429},
title = {A predictive view of Bayesian clustering},
volume = {136},
journal = {Journal of Statistical Planning and Inference},
doi = {10.1016/j.jspi.2004.09.015}
}
```

```BibTeX
@article{data,
author = {Singh, Abhilash and Kotiyal, Vaibhav and Sharma, Sandeep and Nagar, Jaiprakash and Lee, Cheng-Chi},
year = {2020},
month = {11},
pages = {208253 - 208263},
title = {A Machine Learning Approach to Predict the Average Localization Error With Applications to Wireless Sensor Networks},
volume = {8},
journal = {IEEE Access},
doi = {10.1109/ACCESS.2020.3038645}
}
```