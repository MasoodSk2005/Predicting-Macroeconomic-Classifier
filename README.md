# Predicting Macroeconomics Classification and Analysing Development Indices

This project examines whether a country's income group can be classified using a set of economic and human-development indicators. It also compares how the Random Forest model behaves when selected income-related features are removed.

Rather than treating one model as the final answer, the notebook runs three feature configurations using the same data split and evaluation process. This makes it easier to see which results change when GDP per capita and GNI per capita are excluded.

## What the project does

The notebook:

- combines World Bank country metadata and 2023 GDP-per-capita data with UNDP Human Development Index (HDI) indicators;
- cleans and joins the datasets using country codes and country names;
- labels countries as **High income** or **Not high income**;
- explores the resulting data and class distribution;
- trains and tunes three Random Forest classifiers;
- compares the models using held-out test metrics, confusion matrices, ROC curves, and feature-importance plots; and
- displays a sample of country-level predictions with the model's estimated probability of being in the High income class.

## The three experiments

All three models use the same stratified train-test split and the same preprocessing and tuning setup. Only their input features differ.

| Experiment | Features used |
|---|---|
| All 6 Features | GDP per capita (2023), HDI value, life expectancy, expected years of schooling, mean years of schooling, and GNI per capita |
| Without GNI | All of the above except GNI per capita |
| Without GDP & GNI | HDI value, life expectancy, expected years of schooling, and mean years of schooling |

The third experiment is not completely free of income information. HDI includes a GNI-related component, so removing the two standalone GDP/GNI columns does not remove every income signal.

## Data sources and required files

The notebook expects these three datasets:

- `Metadata.csv` — World Bank country metadata, including country codes, names, and income groups.
- `GDP.csv` — GDP-per-capita data containing the `2023` column.
- `HDI.xlsx` — UNDP HDI data, including the worksheet named `Table 1. HDI`.

The notebook searches for the files either in the current working directory or in the `PMC/Datasets/` subfolder. Keep the filenames and sheet name as expected by the notebook.

The datasets are not bundled with this README. Place the required files in one of the supported locations before running the notebook.

## Method

### Data preparation

The notebook joins the metadata and GDP datasets using country codes, then joins the resulting table with HDI data using country names. It cleans the numeric indicator columns, converts them to numeric values, and drops rows that are missing any of the six original features or the income-group label.

The target is defined as:

- `1` — High income
- `0` — Not high income

### Training and evaluation

- Train-test split: 80% training and 20% testing
- Stratification: preserves the class proportions in both subsets
- Random state: `42`
- Cross-validation: 5-fold stratified cross-validation on the training data
- Model: Random Forest Classifier
- Model selection: `GridSearchCV`, using F1-score
- Class weighting: `balanced`

The preprocessing steps are part of the scikit-learn pipeline: missing values are imputed with the median and numeric features are standard-scaled. The notebook searches the following Random Forest parameters:

| Parameter | Values tested |
|---|---|
| `n_estimators` | 100, 250 |
| `max_depth` | None, 5, 10 |
| `min_samples_leaf` | 1, 2 |

The test set is held back for final evaluation of each selected model, rather than used for hyperparameter selection.

## Evaluation

The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrices
- ROC curves
- Random Forest feature importance

It also compares the reproduced six-feature model with the results recorded in the project's documentation.

## Results recorded in the notebook

The run documented in the notebook used 164 countries, with 131 rows in the training set and 33 in the test set.

| Metric | All 6 Features | Without GNI | Without GDP & GNI |
|---|---:|---:|---:|
| Accuracy | 96.97% | 96.97% | 90.91% |
| Precision | 100.00% | 100.00% | 90.91% |
| Recall | 91.67% | 91.67% | 83.33% |
| F1-score | 95.65% | 95.65% | 86.96% |
| ROC-AUC | 94.25% | 93.65% | 95.24% |

These figures are the results recorded for one train-test split, not a guarantee of performance on other datasets or future years. Removing GNI alone did not change accuracy, precision, recall, or F1 in this run, although ROC-AUC changed slightly. Removing both GDP and GNI reduced the threshold-based metrics, while ROC-AUC increased. The metrics describe different aspects of performance, so they do not necessarily move together.

## Important interpretation and limitations

The target income group comes from World Bank metadata, while GDP per capita and GNI per capita are closely related to income-group definitions. This creates a target-proxy dependence and potential leakage concern: the model may be learning indicators closely tied to the way the target is defined, rather than discovering an independent relationship.

The feature-removal experiments help explore that dependence, but they do not prove that the remaining features independently predict income classification. HDI also retains an income-related component. The dataset is relatively small, and a handful of changed predictions can noticeably affect the metrics.

For stronger evidence, evaluate the models across multiple stratified splits or validate on a separate dataset from a later year. Avoid treating a small difference on this single split as proof that one feature configuration generalizes better.

## Requirements

Python or newer is recommended.

Install the libraries used in the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Running the notebook

1. Clone or download the project files.
2. Place `Metadata.csv`, `GDP.csv`, and `HDI.xlsx` in the project folder, or put them in `PMC/Datasets/`.
3. Install the dependencies listed above.
4. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
5. Run the cells in order, starting from the imports and data-loading section.

The notebook uses `GridSearchCV` with parallel processing (`n_jobs=-1`), so training may use multiple available CPU cores.

## Project structure

A simple folder layout that works with the notebook:

```text
project/
├── Predicting_Macroeconomics_Classification_and_Analysing_Development_Indices.ipynb
├── Metadata.csv
├── GDP.csv
└── HDI.xlsx
```

Alternatively, the datasets can be stored under `PMC/Datasets/` as supported by the data-loading cell.

## Libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- OpenPyXL

## Purpose

This project is an experiment in country-level classification and feature analysis. Its main focus is comparing model performance under different feature sets and interpreting those differences carefully—not making causal claims about economic development or predicting future income-group changes.
