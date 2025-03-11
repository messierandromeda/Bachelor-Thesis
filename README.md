# Bachelor Thesis: Drug-Drug GNN for Drug Combination Prediction and Explanations
## Summary
This repository contains my bachelor thesis "Drug-Drug GNN for Drug Combination Prediction and Explanations", data visualizations and the related code. The cardiovascular dataset used in my thesis is based on the distance measures proposed by [Network-based prediction of drug combinations](https://www.nature.com/articles/s41467-019-09186-x) by Cheng et al (2019) and [CombDNF](https://www.biorxiv.org/content/10.1101/2025.02.18.637825v1), an abbreviation for **Comb**inations with **D**isease-specificity **N**etwork-based **F**eatures, where the code implementation can be found [here](https://github.com/DILiS-lab/CombDNF). In short, drug combinations can be viewed as a binary edge classification problem, where the labels are 0 (adverse) or 1 (effective) with the nodes and edges representing the drug-disease features and drug-drug features respectively. For more detailed information about the code, see the repository structure section.

## Packages used: 
python=3.12.2, torch=2.5.1, torch-geometric=2.5.3, pandas=2.2.2, numpy=1.26.4, matplotlib=3.9.1, scikit-learn=1.5.1, seaborn=0.13.2, shap=0.46.0

## Setup
1. Clone the repository
```
$ git clone git@github.com:messierandromeda/Bachelor-Thesis.git
$ cd Bachelor-Thesis
```

2. Optional: create a virtual environment
```
$ python -m venv virtualenv
$ source virtualenv/bin/activate
```

3. Install the required packages
```
$ pip install -r requirements.txt
```

## Data format
Below are some synthetic data to show the file structure used that are used for the training the models:

### Baselines
```baseline_features.csv```
| drugcomb_sorted	| drugA |	drugB |	edge_feature_1 | ... | edge_feature_n | node_feature_1_drugA | ... | node_feature_m_drugA | node_feature_1_drugB | ... | node_feature_m_drugB | label | 
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| DB00001_DB00002 | 1 | 2 | 0.91 | ... | 2.83 | 0.15 | ... | 0.23 | 0.32 | ... | 0.52 | 0
| DB00001_DB00002 | 2 | 1 | 0.91 | ... | 2.83 | 0.32 | ... | 0.52 | 0.15 | ... | 0.23 | 0
| DB00003_DB00004 | 3 | 4 | 0.62 | ... | 0.33 | 0.17 | ... | 0.21 | 0.12 | ... | 0.54 | 0
| DB00003_DB00004 | 4 | 3 | 0.62 | ... | 0.33 | 0.12 | ... | 0.54 | 0.17 | ... | 0.21 | 0
| DB00005_DB00006 | 5 | 6 | 0.78 | ... | 0.99 | 0.12 | ... | 0.77 | 0.01 | ... | 0.06 | 1
| DB00005_DB00006 | 6 | 5 | 0.78 | ... | 0.99 | 0.01 | ... | 0.06 | 0.12 | ... | 0.77 | 1

### GNN Models
```gnn_edge_features.csv```:
| drugcomb_sorted | drugA | drugA_ID | drugB | drugB_ID | feature_1 | ... | feature_n | label
| -------- | ------- | ------- |  ------- | ------- | ------- | ------- | ------- | ------- | 
| DB000010_DB000020 |  10  | 1 |  20 | 2 | 0.5 | ... | 0.8 | 0 |
| DB000010_DB000020 |  20  | 2 |  10 | 1 | 0.5 | ... | 0.8 | 0 |
| DB000020_DB000030 |  20    | 2 | 30 | 3 | 0.1 | ... | 0.41 | 1 | 
| DB000020_DB000030 |  30    | 3 | 20 | 2 | 0.1 | ... | 0.41 | 1 | 
| DB000030_DB000040 | 30 | 3 | 40 | 4 | 0.26 | ... | 0.48 | 0 |
| DB000030_DB000040 | 40 | 4 | 30 | 3 | 0.26 | ... | 0.48 | 0 |

```gnn_node_features.csv```:
| ID | drug | feature_1 | ... | feature_n |
| -------- | ------- | ------- |  ------- | ------- |
| 0 | 1 | 0.145 | ... | 1.56 | 
| 1 | 6 | 2.46 | ... | 2.31 | 
| 2 | 15 | 1.1 | ... | 0.81 |

## Repository structure
There are the following folders:
- ``Drug_combination_data``: the original datasets 
- ``data``: the data used for the prediction tasks after data preprocessing
- ``code``: all the jupyter notebooks / python files for data preprocessing, visualization, training and evaluation 
- ``visualizations``: visualizing the values for the node and edge features in several graphs
- ``explainability``: the shapley value visualizations for the random forest and XGBoost models
- ``evaluation``: the MCC, accuracy, precision and recall scores obtained for each model (Random forest, XGBoost, Neural Network, GCN, GAT, Graph Transformer)

The following jupyter notebooks are:
- Node and edge feature selection: ``Feature_selection.ipynb``
  -  The new version of ``Data_Preprocessing_Visualization_Final.ipynb``, where the code is reorganized into the functions below:
    - ``preprocess_data`` takes the 4 files given, and processes them into the required format needed to extract the features.
      1. DrugCombiNet_drug_disease_scores.tsv
      2. DrugCombiNet_drug_disease_z_scores.tsv
      3. DrugCombiNet_drug_drug_scores.tsv
      4. groundtruth_cardiovascular_2drugs_only.tsv
    - The drug IDs are converted into integers
    - ``baseline_data`` combines the selected node and edge features together into 1 csv file: baseline_features.csv
    - ``gnn_data`` creates 1 csv file for the selected node features and 1 csv file for the selected edge features: gnn_node_features.csv, gnn_edge_features.csv

- Visualizing the node, edge features and labels: ``Data Visualization.ipynb``

- Data preprocessing: ``Data_Preprocessing_Visualization_Final.ipynb``
  - Data preprocessing by combining different csv files
  - Converting the data into the ``Data`` format in PyTorch Geometric
  - Data visualization of the node and edge features from the cardiovascular dataset
  - Ensure no data leakage by checking the combined drug ID (assumed that the drug ID is consistent)
  - Mapped the drug IDs to sequential numbers to match the dimensions
  - The data for the neural network is duplicated so that drugA - drugB have the same label as drugB - drugA
  
- Baselines: ``Baseline_RF_XGBoost.ipynb`` and ``Baseline_NN_Final.ipynb``
  - Used scikit-learn for random forest and xgboost and PyTorch for the neural network
  - 10-fold cross-validation with StratifiedGroupKFold with train/validation/test set
  - Hyperparameter tuning: used best validation result as the final model parameters with slight changes
  - Explanation done with feature importance and shapley values (``shap.TreeExplainer``)

- Graph architectures: ``GNN.ipynb``
  - Contains 3 different types of graph architectures: GCN, GAT, Graph Transformer
  - Used weighted cross-entropy loss to ensure that not all samples are being classified as 0
  - 10-fold cross-validation with StratifiedGroupKFold with the same folds as the baselines with train/validation/test set
  - Hyperparameter tuning: learning rate, weight decay, value of the weighted cross entropy loss
  - Evaluated the accuracy using the Matthews Correlation Coefficient due to imbalanced data
  - Implemented a Graph Convolutional Network (GCN) with node features and edge indices
    - Edge classification is implemented by altering the original function so that the output are not the node features, but the probability that the edge belongs to class 0 or 1
  - Implemented a Graph Attention Network (GAT) and Graph Transformer with node features, edge indices and edge features
 
- Visualizing the results: ``Results.ipynb``
  - Compares all the baseline and graph-based architectures with the following metrics: Matthews Correlation Coefficient (MCC), accuracy, precision, recall
