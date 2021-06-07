import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:

   def __init__(self):
      self.min_max_scaler = MinMaxScaler()

   def load_dataset(self, csv_data_file_path):
      return pandas.read_csv(csv_data_file_path)

   def sort_dataset_by_column(self, dataset, sorting_column):
      return dataset.sort_values(sorting_column)

   def print_dataset(self, dataset, number_of_rows_to_print):
      print(dataset.head(number_of_rows_to_print))

   def print_dataset_values_of_column(self, dataset, dataset_column):
      for row in range(len(dataset.index)):
         print(dataset[[dataset_column]].values[row])

   def normalize_dataset_values_of_column(self, dataset, dataset_column):
      return self.min_max_scaler.fit_transform(dataset[[dataset_column]].values)

   def print_original_and_normalized_dataset_column(self, dataset, dataset_column, normalized_column):
      for row in range(len(dataset.index)):
         print("ORIGINAL = " + str(dataset[[dataset_column]].values[row]) + " --> NORMALIZED = " + str(normalized_column[row]))

   def check_if_dataset_column_has_null_values(self, dataset, dataset_column):
      return len(dataset.index) != dataset[[dataset_column]].count()[0]

   def dataset_column_null_values_count(self, dataset, dataset_column):
      return len(dataset.index) - dataset[[dataset_column]].count()[0]

   def split_normalized_train_and_test_data_chunks(self, normalized_column, trainning_percent):
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

   def split_normalized_past_and_future_data_chunks(self, normalized_column_chunk, start_index, end_index, past_size, future_size):
      normalized_past_data_chunk = []
      normalized_future_data_chunk = []
      for index in range(start_index + past_size, end_index):
         normalized_past_data_chunk.append(numpy.reshape(normalized_column_chunk[range(index - past_size, index)], (past_size, 1)))
         normalized_future_data_chunk.append(normalized_column_chunk[index + future_size])
      return numpy.array(normalized_past_data_chunk), numpy.array(normalized_future_data_chunk)
