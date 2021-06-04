from DatasetManager import *
from GraphPlotter import *

csv_data_file_path = "bitcoin_test_sample.csv" # Dataset Structure: [Date | Symbol | Open | High | Low | Close | Volume BTC | Volume USD]

datasetManager = DatasetManager()
graphPlotter = GraphPlotter()

## LOAD BITCOIN DATASET
dataset = datasetManager.load_dataset(csv_data_file_path)

## SORT BITCOIN DATASET BY 'Date' COLUMN (ASCENDING MODE)
dataset = datasetManager.sort_dataset_by_column(dataset, 'Date')

## PRINT BITCOIN DATASET's FIRST ROWS (HEAD)
datasetManager.print_dataset(dataset, 10)

## PLOT GRAPH: 'Close' Vs 'Date'
graphPlotter.plot_graph("Historical Bitcoin Price", dataset, 'Close Price (USD)', dataset[['Close']], 'Date', dataset['Date'], 30, 90)

print("Column has null values? " + str(datasetManager.check_if_dataset_column_has_null_values(dataset, 'Close')))
print("Column null values count: " + str(datasetManager.dataset_column_null_values_count(dataset, 'Close')))
