<h1 align="center">
  Predictive view of Bayesian Clustering
  <br/>
</h1>

<p align="center">Project for the course "Bayesian Statistics" at ENSAE Paris.  <br/> </p>

## Introduction

In this project, we implement two methods described related to bayesian clustering in the paper [A predictive view of Bayesian clustering](https://joshuachan.org/papers/AD_ML.pdf) by Fernando A. Quintana.

A short report is associated with this repository, and can be found [here](), describing the methods and the results obtained.

## Files and folders

* The implementation is done in Python, and the code is available in the `src/model` folder.
* The notebooks used to generate the results are available in the `notebook` folder.
    - The notebook `MBC_viz.ipynb` contains results of the application of Model Based Clustering method
    - The notebook `DBC_viz.ipynb` contains results of Dirichlet based clustering method

## Data used

The data set `mcs_ds_edited_iter_shuffled.csv` used in the project is available in the `data` folder and is taken from the [UCI  Machine Learning Repo](https://archive.ics.uci.edu/ml/datasets/Average+Localization+Error+%28ALE%29+in+sensor+node+localization+process+in+WSNs#)

## References

The resources used for the project are the follows: 
```bash
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

```bash
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