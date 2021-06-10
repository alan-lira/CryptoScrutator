import pandas
import numpy
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, median_absolute_error, r2_score, mean_tweedie_deviance
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:

   def __init__(self):
      self.min_max_scaler = MinMaxScaler()

   def loadDataset(self, csv_data_file_path):
      return pandas.read_csv(csv_data_file_path)

   def sortDatasetByColumn(self, dataset, sorting_column):
      return dataset.sort_values(sorting_column)

   def printDataset(self, dataset, number_of_rows_to_print):
      print(dataset.head(number_of_rows_to_print))

   def printDatasetValuesOfColumn(self, dataset, dataset_column):
      for row in range(len(dataset.index)):
         print(dataset[[dataset_column]].values[row])

   def normalizeDatasetValuesOfColumn(self, dataset, dataset_column):
      return self.min_max_scaler.fit_transform(dataset[[dataset_column]].values)

   def inverseTransformData(self, data):
      return self.min_max_scaler.inverse_transform(data)

   def printRealAndNormalizedDatasetColumn(self, dataset, dataset_column, normalized_column):
      for row in range(len(dataset.index)):
         print("Real = " + str(dataset[[dataset_column]].values[row]) + " --> Normalized = " + str(normalized_column[row]))

   def checkIfDatasetColumnHasNullValues(self, dataset, dataset_column):
      return len(dataset.index) != dataset[[dataset_column]].count()[0]

   def datasetColumnNullValuesCount(self, dataset, dataset_column):
      return len(dataset.index) - dataset[[dataset_column]].count()[0]

   def splitNormalizedTrainAndTestDataChunks(self, normalized_column, trainning_percent):
      normalized_train_data_chunk = []
      normalized_test_data_chunk = []
      trainning_chunk_size = int(len(normalized_column) * trainning_percent)
      testing_chunk_size = len(normalized_column) - trainning_chunk_size
      for train_index in range(0, trainning_chunk_size):
         normalized_train_data_chunk.append(normalized_column[train_index])
      for test_index in range(trainning_chunk_size, len(normalized_column)):
         normalized_test_data_chunk.append(normalized_column[test_index])
      normalized_train_data_chunk = numpy.reshape(normalized_train_data_chunk, (trainning_chunk_size, 1))
      normalized_test_data_chunk = numpy.reshape(normalized_test_data_chunk, (testing_chunk_size, 1))
      return normalized_train_data_chunk, normalized_test_data_chunk

   def splitNormalizedPastAndFutureDataChunks(self, normalized_column_chunk, start_index, end_index, learning_size, prediction_size):
      normalized_past_data_chunk = []
      normalized_future_data_chunk = []
      for index in range(start_index + learning_size, end_index):
         normalized_past_data_chunk.append(numpy.reshape(normalized_column_chunk[range(index - learning_size, index)], (learning_size, 1)))
         normalized_future_data_chunk.append(normalized_column_chunk[index + prediction_size])
      return numpy.array(normalized_past_data_chunk), numpy.array(normalized_future_data_chunk)

   def printRegressionMetrics(self, real_values, predicted_values):
      print("Explained Variance Regression Score: " + str(explained_variance_score(real_values, predicted_values)))
      print("Maximum Residual Error: " + str(max_error(real_values, predicted_values)))
      print("Mean Absolute Error: " + str(mean_absolute_error(real_values, predicted_values)))
      print("Mean Square Error: " + str(mean_squared_error(real_values, predicted_values)))
      print("Squared Logarithmic (Quadratic) Error: " + str(mean_squared_log_error(real_values, predicted_values)))
      print("Mean Absolute Percentage Error (Deviation): " + str(mean_absolute_percentage_error(real_values, predicted_values)))
      print("Median Absolute Error: " + str(median_absolute_error(real_values, predicted_values)))
      print("Coefficient of Determination (R²): " + str(r2_score(real_values, predicted_values)))
      print("Mean Poisson Deviance: " + str(mean_tweedie_deviance(real_values, predicted_values, power = 1)))
      print("Mean Gamma Deviance: " + str(mean_tweedie_deviance(real_values, predicted_values, power = 2)))
      print("\n")
