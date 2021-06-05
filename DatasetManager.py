import pandas
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:

   def __init__(self):
      self.min_max_scaler = MinMaxScaler()

   def load_dataset(self, csv_data_file_path):
      return pandas.read_csv(csv_data_file_path)

   def sort_dataset_by_column(self, dataset, dataset_column):
      return dataset.sort_values(dataset_column)

   def print_dataset(self, dataset, number_of_rows):
      print(dataset.head(number_of_rows))

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
