import pandas

class DatasetManager:

   def __init__(self):
      pass

   def load_dataset(self, csv_data_file_path):
      return pandas.read_csv(csv_data_file_path)

   def sort_dataset_by_column(self, dataset, dataset_column):
      return dataset.sort_values(dataset_column)

   def print_dataset(self, dataset, number_of_rows):
      print(dataset.head(number_of_rows))

   def check_if_dataset_column_has_null_values(self, dataset, dataset_column):
      return len(dataset.index) != dataset[[dataset_column]].count()[0]

   def dataset_column_null_values_count(self, dataset, dataset_column):
      return len(dataset.index) - dataset[[dataset_column]].count()[0]
