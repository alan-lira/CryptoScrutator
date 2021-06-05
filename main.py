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

## PRINT INFO ABOUT NULL VALUES AT 'Close' COLUMN (CLOSE PRICE)
print("Column has null values? " + str(datasetManager.check_if_dataset_column_has_null_values(dataset, 'Close')))
print("Column null values count: " + str(datasetManager.dataset_column_null_values_count(dataset, 'Close')))

## PRINT BITCOIN DATASET's 'Close' COLUMN (CLOSE PRICE)
datasetManager.print_dataset_values_of_column(dataset, 'Close')

## NORMALIZE BITCOIN DATASET's 'Close' COLUMN (CLOSE PRICE)
normalized_close_column = datasetManager.normalize_dataset_values_of_column(dataset, 'Close')

## PRINT ORIGINAL AND NORMALIZED VALUES OF 'Close' COLUMN (CLOSE PRICE)
datasetManager.print_original_and_normalized_dataset_column(dataset, 'Close', normalized_close_column)

## SPLIT NORMALIZED TRAIN AND TEST DATA CHUNKS (TRAINNING DATA PERCENT: 80%)
trainning_data_percent = 0.8
normalized_train_data_chunk, normalized_test_data_chunk = datasetManager.split_normalized_train_and_test_data_chunks(normalized_close_column, trainning_data_percent)

## SET AMOUNT OF DAYS TO LEARN TO PREDICT THE FUTURE TIME SERIES
past_size = 5 # LEARN 5 DAYS
future_size = 0 # PREDICT THE NEXT DAY

## SPLIT NORMALIZED PAST AND FUTURE TRAIN DATA CHUNKS
train_start_index = 0
train_end_index = len(normalized_train_data_chunk)
normalized_past_train_data_chunk, normalized_future_train_data_chunk = datasetManager.split_normalized_past_and_future_data_chunks(normalized_train_data_chunk, train_start_index, train_end_index, past_size, future_size)

## SPLIT NORMALIZED PAST AND FUTURE TEST DATA CHUNKS
test_start_index = 0
test_end_index = len(normalized_test_data_chunk)
normalized_past_test_data_chunk, normalized_future_test_data_chunk = datasetManager.split_normalized_past_and_future_data_chunks(normalized_test_data_chunk, test_start_index, test_end_index, past_size, future_size)
