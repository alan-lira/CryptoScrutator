import pandas
import numpy
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, median_absolute_error, r2_score

class DatasetManager:

   def __init__(self):
      pass

   def loadDataset(self, csv_data_file_path):
      return pandas.read_csv(csv_data_file_path)

   def sortDatasetByColumn(self, dataset, sorting_column):
      return dataset.sort_values(sorting_column)

   def getNormalizedTrainAndTestDataChunks(self, normalized_column, trainning_percent):
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

   def getLearningAndPredictionChunksFromNormalizedTrainningChunk(self, normalized_trainning_chunk, start_index, end_index, learning_size, prediction_size):
      normalized_trainning_learning_chunk = []
      normalized_trainning_prediction_chunk = []
      for index in range(start_index + learning_size, end_index):
         normalized_trainning_learning_chunk.append(numpy.reshape(normalized_trainning_chunk[range(index - learning_size, index)], (learning_size, 1)))
         normalized_trainning_prediction_chunk.append(normalized_trainning_chunk[index + prediction_size])
      return numpy.array(normalized_trainning_learning_chunk), numpy.array(normalized_trainning_prediction_chunk)

   def getToPredictChunkFromNormalizedTestingChunk(self, normalized_testing_chunk, start_index, end_index, learning_size):
      normalized_testing_to_predict_chunk = []
      for index in range(start_index + learning_size, end_index):
         normalized_testing_to_predict_chunk.append(numpy.reshape(normalized_testing_chunk[range(index - learning_size, index)], (learning_size, 1)))
      return numpy.array(normalized_testing_to_predict_chunk)

   def getActualValuesToCompareChunk(self, chosen_column, start_index, end_index, learning_size, prediction_size):
      actual_values_to_compare_chunk = []
      for index in range(start_index + learning_size, end_index):
         actual_values_to_compare_chunk.append(chosen_column[index + prediction_size])
      return numpy.array(actual_values_to_compare_chunk)

   def printRegressionMetrics(self, actual_values, predicted_values):
      actual_values[numpy.isnan(actual_values)] = 0
      actual_values[numpy.isinf(actual_values)] = 0
      predicted_values[numpy.isnan(predicted_values)] = 0
      predicted_values[numpy.isinf(predicted_values)] = 0
      print("Explained Variance Score: " + str(explained_variance_score(actual_values, predicted_values)) + " --> Best possible score: 1.0")
      print("Maximum Residual Error: " + str(max_error(actual_values, predicted_values)) + " --> Best possible value: 0.0")
      print("Mean Absolute Error (MAE): " + str(mean_absolute_error(actual_values, predicted_values)) + " --> Best possible value: 0.0")
      print("Mean Squared Error (MSE): " + str(mean_squared_error(actual_values, predicted_values, squared = True)) + " --> Best possible value: 0.0")
      print("Root Mean Squared Error (RMSE): " + str(mean_squared_error(actual_values, predicted_values, squared = False)) + " --> Best possible value: 0.0")
      print("Mean Squared Logarithmic Error (MSLE): " + str(mean_squared_log_error(actual_values, predicted_values)) + " --> Best possible value: 0.0")
      print("Mean Absolute Percentage Error (MAPE): " + str(mean_absolute_percentage_error(actual_values, predicted_values)) + " --> Best possible value: 0.0")
      print("Median Absolute Deviation (MAD): " + str(median_absolute_error(actual_values, predicted_values)) + " --> Best possible value: 0.0")
      print("Coefficient of Determination (R??): " + str(r2_score(actual_values, predicted_values)) + " --> Best possible score: 1.0")
      print("\n")
