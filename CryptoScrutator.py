import csv
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # AVOID «TensorFlow» LOGGING
import string
import time
from DatasetManager import *
from GraphPlotter import *
from NaiveInvestor import *
from RecurrentNeuralNetworkManager import *
from sklearn.preprocessing import MinMaxScaler

class CryptoScrutator:

   def __init__(self):
      self.datasetManager = DatasetManager()
      self.graphPlotter = GraphPlotter()
      self.naiveInvestor = NaiveInvestor()
      self.recurrentNeuralNetworkManager = RecurrentNeuralNetworkManager()
      self.min_max_scaler = MinMaxScaler()
      self.print_metrics_history_boolean = None
      self.plot_metrics_graphs_boolean = None
      self.plot_actual_values_vs_predicted_values_graph_boolean = None
      self.print_regression_metrics_boolean = None
      self.plot_all_rnn_models_comparison_graph_boolean = None
      self.cryptocoin_name = None
      self.dataset_file = None
      self.sorting_column = None
      self.chosen_column = None
      self.cryptocoin_dataset = None
      self.chosen_column_data_actual_values_to_compare_chunk = None
      self.normalized_chosen_column_data = None
      self.trainning_data_percent = None
      self.normalized_trainning_data_chunk = None
      self.normalized_trainning_data_learning_chunk = None
      self.normalized_trainning_data_prediction_chunk = None
      self.normalized_testing_data_chunk = None
      self.normalized_testing_data_to_predict_chunk = None
      self.learning_size = None
      self.prediction_size = None
      self.model_name = None
      self.number_of_simplernn_units = None
      self.number_of_lstm_units = None
      self.number_of_gru_units = None
      self.activation = None
      self.recurrent_initializer = None
      self.recurrent_activation = None
      self.number_of_time_steps = None
      self.number_of_features = None
      self.input_shape = None
      self.negative_slope_coefficient = None
      self.fraction_of_the_input_units_to_drop = None
      self.number_of_dense_units = None
      self.loss_function_name = None
      self.optimizer_name = None
      self.optimizer_learning_rate = None
      self.metrics_list = None
      self.validation_split_percent = None
      self.number_of_epochs = None
      self.batch_size = None
      self.shuffle_boolean = None
      self.rnn_model_type = None
      self.prediction_history = None
      self.simplernn_predicted_values = None
      self.lstm_predicted_values = None
      self.gru_predicted_values = None
      self.initial_balance_in_usd = None
      self.initial_balance_in_bitcoin = None
      self.selling_bitcoin_strategy_percent = None
      self.buying_bitcoin_strategy_percent = None

   def loadCryptoScrutatorSettings(self, crypto_scrutator_settings_file):
      with open(crypto_scrutator_settings_file, mode = "r") as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "print_metrics_history_boolean":
                  self.print_metrics_history_boolean = value == "True"
               elif key == "plot_metrics_graphs_boolean":
                  self.plot_metrics_graphs_boolean = value == "True"
               elif key == "plot_actual_values_vs_predicted_values_graph_boolean":
                  self.plot_actual_values_vs_predicted_values_graph_boolean = value == "True"
               elif key == "print_regression_metrics_boolean":
                  self.print_regression_metrics_boolean = value == "True"
               elif key == "plot_all_rnn_models_comparison_graph_boolean":
                  self.plot_all_rnn_models_comparison_graph_boolean = value == "True"

   def _getDatasetFileDataHeader(self):
      dataset_file_header = []
      with open(self.dataset_file, mode = "r") as to_get:
         csvReader = csv.reader(to_get)
         dataset_file_header = next(csvReader)
      return dataset_file_header

   def loadDatasetSettings(self, dataset_settings_file):
      with open(dataset_settings_file, mode = "r") as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "cryptocoin_name":
                  self.cryptocoin_name = str(value)
               elif key == "dataset_file":
                  dataset_file = str(value)
                  dataset_file_exists = os.path.isfile(dataset_file)
                  if dataset_file_exists:
                     self.dataset_file = dataset_file
                  else:
                     print("Dataset '" + dataset_file + "' not found!")
                     raise SystemExit
               elif key == "sorting_column":
                  sorting_column = str(value)
                  dataset_file_header = self._getDatasetFileDataHeader()
                  try:
                     column_to_verify_index = dataset_file_header.index(sorting_column)
                     self.sorting_column = sorting_column
                  except ValueError:
                     print("Dataset '" + self.dataset_file + "' does not have '" + sorting_column + "' column...\nSet one of the below as the 'sorting_column' parameter:")
                     for header in dataset_file_header:
                        print(header)
                     raise SystemExit
               elif key == "chosen_column":
                  chosen_column = str(value)
                  dataset_file_header = self._getDatasetFileDataHeader()
                  try:
                     column_to_verify_index = dataset_file_header.index(chosen_column)
                     self.chosen_column = chosen_column
                  except ValueError:
                     print("Dataset '" + self.dataset_file + "' does not have '" + chosen_column + "' column...\nSet one of the below as the 'chosen_column' parameter:")
                     for header in dataset_file_header:
                        print(header)
                     raise SystemExit

   def loadRNNModelHyperparametersSettings(self, rnn_model_hyperparameters_settings_file):
      with open(rnn_model_hyperparameters_settings_file, mode = "r") as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "model_name":
                  self.model_name = str(value)
               elif key == "trainning_data_percent":
                  trainning_data_percent = float(value)
                  if 0 < trainning_data_percent < 1:
                     self.trainning_data_percent = trainning_data_percent
                  else:
                     print("'trainning_data_percent' parameter must lie between 0 and 1 (E.g., 0.8).")
                     raise SystemExit
               elif key == "learning_size":
                  learning_size = int(value)
                  if learning_size > 0:
                     self.learning_size = learning_size
                  else:
                     print("'learning_size' parameter must be higher than 0 (learn X values in the time series, X > 0).")
                     raise SystemExit
               elif key == "prediction_size":
                  prediction_size = int(value)
                  if prediction_size == 0:
                     self.prediction_size = prediction_size
                  else:
                     print("'prediction_size' parameter must be 0 (predict the next value in the time series).")
                     raise SystemExit
               elif key == "number_of_simplernn_units":
                  number_of_simplernn_units = int(value)
                  if number_of_simplernn_units > 0:
                     self.number_of_simplernn_units = number_of_simplernn_units
                  else:
                     print("'number_of_simplernn_units' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "number_of_lstm_units":
                  number_of_lstm_units = int(value)
                  if number_of_lstm_units > 0:
                     self.number_of_lstm_units = number_of_lstm_units
                  else:
                     print("'number_of_lstm_units' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "number_of_gru_units":
                  number_of_gru_units = int(value)
                  if number_of_gru_units > 0:
                     self.number_of_gru_units = number_of_gru_units
                  else:
                     print("'number_of_gru_units' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "activation":
                  self.activation = str(value)
               elif key == "recurrent_initializer":
                  self.recurrent_initializer = str(value)
               elif key == "recurrent_activation":
                  self.recurrent_activation = str(value)
               elif key == "number_of_time_steps":
                  if value == "learning_size":
                     self.number_of_time_steps = self.learning_size
                  else:
                     number_of_time_steps = int(value)
                     if number_of_time_steps > 0:
                        self.number_of_time_steps = number_of_time_steps
                     else:
                        print("'number_of_time_steps' parameter must be higher than 0.")
                        raise SystemExit
               elif key == "number_of_features":
                  number_of_features = int(value)
                  if number_of_features == 1:
                     self.number_of_features = number_of_features
                  else:
                     print("'number_of_features' parameter must be 1 (use just one feature).")
                     raise SystemExit
               elif key == "input_shape":
                  if value == "(number_of_time_steps, number_of_features)":
                     self.input_shape = (self.number_of_time_steps, self.number_of_features)
                  else:
                     self.input_shape = value
               elif key == "negative_slope_coefficient":
                  negative_slope_coefficient = float(value)
                  if negative_slope_coefficient >= 0:
                     self.negative_slope_coefficient = negative_slope_coefficient
                  else:
                     print("'negative_slope_coefficient' parameter must be greater than or equal to 0.")
                     raise SystemExit
               elif key == "fraction_of_the_input_units_to_drop":
                  fraction_of_the_input_units_to_drop = float(value)
                  if 0 < fraction_of_the_input_units_to_drop < 1:
                     self.fraction_of_the_input_units_to_drop = fraction_of_the_input_units_to_drop
                  else:
                     print("'fraction_of_the_input_units_to_drop' parameter must lie between 0 and 1 (E.g., 0.1).")
                     raise SystemExit
               elif key == "number_of_dense_units":
                  number_of_dense_units = int(value)
                  if number_of_dense_units > 0:
                     self.number_of_dense_units = number_of_dense_units
                  else:
                     print("'number_of_dense_units' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "loss_function_name":
                  self.loss_function_name = str(value)
               elif key == "optimizer_name":
                  self.optimizer_name = str(value)
               elif key == "optimizer_learning_rate":
                  optimizer_learning_rate = float(value)
                  if optimizer_learning_rate > 0:
                     self.optimizer_learning_rate = optimizer_learning_rate
                  else:
                     print("'optimizer_learning_rate' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "metrics_list":
                  self.metrics_list = value.split(", ")
               elif key == "validation_split_percent":
                  validation_split_percent = float(value)
                  if 0 < validation_split_percent < 1:
                     self.validation_split_percent = validation_split_percent
                  else:
                     print("'validation_split_percent' parameter must lie between 0 and 1 (E.g., 0.1).")
                     raise SystemExit
               elif key == "number_of_epochs":
                  number_of_epochs = int(value)
                  if number_of_epochs > 0:
                     self.number_of_epochs = number_of_epochs
                  else:
                     print("'number_of_epochs' parameter must be higher than 0.")
                     raise SystemExit
               elif key == "batch_size":
                  if value == "learning_size":
                     self.batch_size = self.learning_size
                  else:
                     batch_size = int(value)
                     if batch_size > 0:
                        self.batch_size = batch_size
                     else:
                        print("'batch_size' parameter must be higher than 0.")
                        raise SystemExit
               elif key == "shuffle_boolean":
                  self.shuffle_boolean = value == "True"

   def loadInvestmentSimulatorSettings(self, investment_simulator_settings_file):
      with open(investment_simulator_settings_file, mode = "r") as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "initial_balance_in_usd":
                  initial_balance_in_usd = float(value)
                  if initial_balance_in_usd >= 0:
                     self.initial_balance_in_usd = initial_balance_in_usd
                     self.naiveInvestor.setBalanceInUSD(self.initial_balance_in_usd)
                  else:
                     print("'initial_balance_in_usd' parameter must be greater than or equal to 0.")
                     raise SystemExit
               elif key == "initial_balance_in_bitcoin":
                  initial_balance_in_bitcoin = float(value)
                  if initial_balance_in_bitcoin >= 0:
                     self.initial_balance_in_bitcoin = initial_balance_in_bitcoin
                     self.naiveInvestor.setBalanceInBitcoin(self.initial_balance_in_bitcoin)
                  else:
                     print("'initial_balance_in_bitcoin' parameter must be greater than or equal to 0.")
                     raise SystemExit
               elif key == "selling_bitcoin_strategy_percent":
                  selling_bitcoin_strategy_percent = float(value)
                  if 0 <= selling_bitcoin_strategy_percent <= 1:
                     self.selling_bitcoin_strategy_percent = selling_bitcoin_strategy_percent
                     self.naiveInvestor.setSellingBitcoinStrategyPercent(self.selling_bitcoin_strategy_percent)
                  else:
                     print("'selling_bitcoin_strategy_percent' parameter must lie between 0 and 1, inclusive (E.g., 0.8).")
                     raise SystemExit
               elif key == "buying_bitcoin_strategy_percent":
                  buying_bitcoin_strategy_percent = float(value)
                  if 0 <= buying_bitcoin_strategy_percent <= 1:
                     self.buying_bitcoin_strategy_percent = buying_bitcoin_strategy_percent
                     self.naiveInvestor.setBuyingBitcoinStrategyPercent(self.buying_bitcoin_strategy_percent)
                  else:
                     print("'buying_strategy_percent' parameter must lie between 0 and 1, inclusive (E.g., 0.4).")
                     raise SystemExit

   def loadCryptocoinDatasetCSV(self):
      self.cryptocoin_dataset = self.datasetManager.loadDataset(self.dataset_file)

   def sortCryptocoinDatasetBySortingColumn(self):
      self.cryptocoin_dataset = self.datasetManager.sortDatasetByColumn(self.cryptocoin_dataset, self.sorting_column)

   def _verifyDatasetFileColumnHasNullData(self, column_to_verify):
      with open(self.dataset_file, mode = "r") as to_verify:
         csvReader = csv.reader(to_verify)
         dataset_file_header = next(csvReader)
         column_to_verify_index = dataset_file_header.index(column_to_verify)
         for dataset_file_row in csvReader:
            if dataset_file_row[column_to_verify_index] == "":
               return True
      return False

   def _setDatasetFile(self, dataset_file):
      self.dataset_file = dataset_file

   def handleDatasetFileChosenAndSortingColumnsNullData(self):
      chosen_column_has_null_data = self._verifyDatasetFileColumnHasNullData(self.chosen_column)
      sorting_column_has_null_data = self._verifyDatasetFileColumnHasNullData(self.sorting_column)
      if chosen_column_has_null_data or sorting_column_has_null_data:
         dataset_file_path = "datasets/"
         dataset_file_extension = ".csv"
         dataset_file_name = self.dataset_file.split(dataset_file_path)[1].split(dataset_file_extension)[0]
         dataset_new_file_name = dataset_file_path + dataset_file_name + "_handled" + dataset_file_extension
         with open(dataset_new_file_name, mode = "w") as handled:
            csvWriter = csv.writer(handled, delimiter = ",")
            with open(self.dataset_file, mode = "r") as to_handle:
               csvReader = csv.reader(to_handle)
               dataset_file_header = next(csvReader)
               csvWriter.writerow(dataset_file_header)
               chosen_column_index = dataset_file_header.index(self.chosen_column)
               sorting_column_index = dataset_file_header.index(self.sorting_column)
               for dataset_file_row in csvReader:
                  if dataset_file_row[chosen_column_index] != "" and dataset_file_row[sorting_column_index] != "":
                     csvWriter.writerow(dataset_file_row)
         self._setDatasetFile(dataset_new_file_name)

   def _getDatasetFileDataLinesCount(self):
      lines_counter = 0
      with open(self.dataset_file, mode = "r") as to_count:
         csvReader = csv.reader(to_count)
         dataset_file_header = next(csvReader)
         lines_counter = len(list(csvReader))
      return lines_counter

   def _calculateDataRowsAddNeeding(self):
      dataset_data_lines_count = self._getDatasetFileDataLinesCount()
      data_rows_to_add = 0
      while True:
         data_rows_to_add = data_rows_to_add + 1
         if (dataset_data_lines_count + data_rows_to_add) - int((dataset_data_lines_count + data_rows_to_add) * self.trainning_data_percent) > self.learning_size:
            break
      return data_rows_to_add

   def _calculateLearningSizeDecreaseNeeding(self):
      dataset_data_lines_count = self._getDatasetFileDataLinesCount()
      valid_learning_size = self.learning_size
      while True:
         valid_learning_size = valid_learning_size - 1
         if (dataset_data_lines_count) - int(dataset_data_lines_count * self.trainning_data_percent) > valid_learning_size:
            break
      return valid_learning_size

   def verifyDatasetFileHasEnoughDataToBePartitioned(self):
      dataset_data_lines_count = self._getDatasetFileDataLinesCount()
      trainning_chunk_size = int(dataset_data_lines_count * self.trainning_data_percent)
      testing_chunk_size = dataset_data_lines_count - trainning_chunk_size
      if testing_chunk_size <= self.learning_size:
         print("Dataset '" + self.dataset_file + "' does not have enough data to be partitioned with the setted parameters...")
         data_rows_to_add = self._calculateDataRowsAddNeeding()
         print("Solution 1: Add " + str(data_rows_to_add) + " more row(s) of data into '"+self.dataset_file+"'s dataset. That is, increasing to " + str(data_rows_to_add + dataset_data_lines_count) + " total rows!")
         valid_learning_size = self._calculateLearningSizeDecreaseNeeding()
         print("Solution 2: Decrease your 'learning_size' to " + str(valid_learning_size) + ". That is, reducing by " + str(self.learning_size - valid_learning_size) + "!")
         raise SystemExit

   def normalizeChosenColumnData(self):
      values_to_normalize = self.cryptocoin_dataset[[self.chosen_column]].values
      self.normalized_chosen_column_data = self.min_max_scaler.fit_transform(values_to_normalize)

   def splitNormalizedChosenColumnDataBetweenTrainningAndTestingChunks(self):
      self.normalized_trainning_data_chunk, self.normalized_testing_data_chunk = self.datasetManager.getNormalizedTrainAndTestDataChunks(self.normalized_chosen_column_data, self.trainning_data_percent)

   def splitNormalizedTrainningChunkBetweenLearningAndPredictionChunks(self):
      train_start_index = 0
      train_end_index = len(self.normalized_trainning_data_chunk)
      self.normalized_trainning_data_learning_chunk, self.normalized_trainning_data_prediction_chunk = self.datasetManager.getLearningAndPredictionChunksFromNormalizedTrainningChunk(self.normalized_trainning_data_chunk, train_start_index, train_end_index, self.learning_size, self.prediction_size)

   def setNormalizedTestingDataToPredictChunk(self):
      test_start_index = 0
      test_end_index = len(self.normalized_testing_data_chunk)
      self.normalized_testing_data_to_predict_chunk = self.datasetManager.getToPredictChunkFromNormalizedTestingChunk(self.normalized_testing_data_chunk, test_start_index, test_end_index, self.learning_size)

   def setChosenColumnDataActualValuesToCompareChunk(self):
      chosen_column_data = self.cryptocoin_dataset[[self.chosen_column]].values
      trainning_chunk_size = int(len(chosen_column_data) * self.trainning_data_percent)
      testing_chunk_size = len(chosen_column_data) - trainning_chunk_size
      start_index = trainning_chunk_size
      end_index = start_index + testing_chunk_size
      self.chosen_column_data_actual_values_to_compare_chunk = self.datasetManager.getActualValuesToCompareChunk(chosen_column_data, start_index, end_index, self.learning_size, self.prediction_size)

   def denormalizeValues(self, normalized_values):
      return self.min_max_scaler.inverse_transform(normalized_values)

   def setRNNModelType(self, rnn_model_type):
      self.rnn_model_type = rnn_model_type

   def _createRNNModel(self):
      ## CREATE «rnn_model_type» RNN MODEL
      self.recurrentNeuralNetworkManager.createEmptySequentialModel(self.model_name+"_"+self.rnn_model_type)

      if self.rnn_model_type == "SimpleRNN":
         ## ADD «SimpleRNN» LAYER
         self.recurrentNeuralNetworkManager.addSimpleRecurrentNeuralNetworkLayer(self.number_of_simplernn_units,
                                                                                 self.activation,
                                                                                 self.recurrent_initializer,
                                                                                 self.input_shape)   
      elif self.rnn_model_type == "LSTM":
         ## ADD «LSTM» LAYER
         self.recurrentNeuralNetworkManager.addLongShortTermMemoryLayer(self.number_of_lstm_units,
                                                                        self.activation,
                                                                        self.recurrent_activation,
                                                                        self.input_shape)
      elif self.rnn_model_type == "GRU":
         ## ADD «GRU» LAYER
         self.recurrentNeuralNetworkManager.addGatedRecurrentUnitLayer(self.number_of_gru_units,
                                                                       self.activation,
                                                                       self.recurrent_activation,
                                                                       self.input_shape)

      ## ADD «LeakyReLu» LAYER
      self.recurrentNeuralNetworkManager.addLeakyRectifiedLinearUnitLayer(self.negative_slope_coefficient)

      ## ADD «Dropout» LAYER
      self.recurrentNeuralNetworkManager.addDropoutLayer(self.fraction_of_the_input_units_to_drop)

      ## ADD «Dense» LAYER
      self.recurrentNeuralNetworkManager.addDenseLayer(self.number_of_dense_units)

      ## LOAD LOSS FUNCTION
      self.recurrentNeuralNetworkManager.loadLossFunction(self.loss_function_name)

      ## LOAD OPTIMIZER
      self.recurrentNeuralNetworkManager.loadOptimizer(self.optimizer_name, self.optimizer_learning_rate)

      ## LOAD METRICS
      self.recurrentNeuralNetworkManager.loadMetrics(self.metrics_list)

      ## COMPILE RNN MODEL
      self.recurrentNeuralNetworkManager.compileSequentialModel()

      ## SUMMARIZE RNN MODEL
      self.recurrentNeuralNetworkManager.summarizeModel()

   def _trainRNNModel(self):
      ## TRAIN RNN MODEL
      self.recurrentNeuralNetworkManager.trainModel(self.normalized_trainning_data_learning_chunk,
                                                    self.normalized_trainning_data_prediction_chunk,
                                                    self.validation_split_percent,
                                                    self.number_of_epochs,
                                                    self.batch_size,
                                                    self.shuffle_boolean)

   def _printTrainnedRNNModelMetricsHistory(self):
      if self.print_metrics_history_boolean == True:
         ## PRINT TRAINNED RNN MODEL METRICS HISTORY
         self.recurrentNeuralNetworkManager.printTrainnedModelMetricsHistory()

   def _plotTrainnedRNNModelMetricsGraphs(self):
      if self.plot_metrics_graphs_boolean == True:
         ## PLOT GRAPH: (X = «Number of Epochs», Y = «Training Metric» Vs «Validation Metric»)
         trainned_model_metrics_history = self.recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
         metrics_to_plot = ["loss"]
         metrics_to_plot.extend(self.metrics_list)
         for metric in metrics_to_plot:
            metric_name = metric.replace("_", " ").title()
            training_metric = trainned_model_metrics_history[metric]
            validation_metric = trainned_model_metrics_history["val_"+metric]
            graph_title = metric_name + " Graph (" + self.rnn_model_type + ")"
            y_label = metric_name
            x_label = "Number of Epochs"
            x_ticks_size = self.number_of_epochs
            curves_labels = ["Training " + metric_name, "Validation " + metric_name]
            curves_colors = ["indianred", "olivedrab"]
            curves_datas = [training_metric, validation_metric]
            self.graphPlotter.plotCurvesGraph(graph_title,
                                              y_label,
                                              x_label,
                                              x_ticks_size,
                                              curves_labels,
                                              curves_colors,
                                              curves_datas)

   def _predictWithTrainnedModel(self):
      ## PREDICT WITH TRAINNED RNN MODEL
      self.recurrentNeuralNetworkManager.predictWithTrainnedModel(self.normalized_testing_data_to_predict_chunk)
      self.prediction_history = self.recurrentNeuralNetworkManager.getPredictionHistory()

   def _denormalizePredictedValues(self):
      ## DENORMALIZE PREDICTED VALUES
      denormalized_predicted_values = self.denormalizeValues(self.prediction_history)
      if self.rnn_model_type == "SimpleRNN":
         self.simplernn_predicted_values = denormalized_predicted_values
      elif self.rnn_model_type == "LSTM":
         self.lstm_predicted_values = denormalized_predicted_values
      elif self.rnn_model_type == "GRU":
         self.gru_predicted_values = denormalized_predicted_values

   def _plotActualValuesPredictedValuesGraphComparison(self):
      ## PLOT ACTUAL VALUES VS PREDICTED VALUES GRAPH
      if self.plot_actual_values_vs_predicted_values_graph_boolean == True:
         ## PLOT GRAPH: (X = 'Days', Y = 'Actual Value' Vs 'Predicted Value')
         graph_title = self.cryptocoin_name + " Predictor"
         y_label = self.chosen_column
         x_label = "Days"
         x_ticks_size = len(self.chosen_column_data_actual_values_to_compare_chunk)
         curves_labels = ["Actual Value"]
         curves_colors = ["gold"]
         curves_datas = [self.chosen_column_data_actual_values_to_compare_chunk]
         if self.rnn_model_type == "SimpleRNN":
            curves_labels.append("Predicted Value ("+self.rnn_model_type+")")
            curves_colors.append("firebrick")
            curves_datas.append(self.simplernn_predicted_values)
         elif self.rnn_model_type == "LSTM":
            curves_labels.append("Predicted Value ("+self.rnn_model_type+")")
            curves_colors.append("lightseagreen")
            curves_datas.append(self.lstm_predicted_values)
         elif self.rnn_model_type == "GRU":
            curves_labels.append("Predicted Value ("+self.rnn_model_type+")")
            curves_colors.append("darkslateblue")
            curves_datas.append(self.gru_predicted_values)
         self.graphPlotter.plotCurvesGraph(graph_title,
                                           y_label,
                                           x_label,
                                           x_ticks_size,
                                           curves_labels,
                                           curves_colors,
                                           curves_datas)

   def _clearRNNModel(self):
      ## CLEAR RNN MODEL
      self.recurrentNeuralNetworkManager.clearModel()

   def _printActualValuesPredictedValuesRegressionMetrics(self):
      if self.print_regression_metrics_boolean == True:
         ## PRINT REGRESSION METRICS
         if self.rnn_model_type == "SimpleRNN":
            self.datasetManager.printRegressionMetrics(self.chosen_column_data_actual_values_to_compare_chunk, self.simplernn_predicted_values)
         elif self.rnn_model_type == "LSTM":
            self.datasetManager.printRegressionMetrics(self.chosen_column_data_actual_values_to_compare_chunk, self.lstm_predicted_values)
         elif self.rnn_model_type == "GRU":
            self.datasetManager.printRegressionMetrics(self.chosen_column_data_actual_values_to_compare_chunk, self.gru_predicted_values)

   def executeRNNModel(self):
      self._createRNNModel()
      startTrainTime = time.time()
      self._trainRNNModel()
      endTrainTime = time.time()
      print("\n" + str(self.rnn_model_type) + " Model Trainning Time: " + str(endTrainTime - startTrainTime) + " seconds.\n")
      self._printTrainnedRNNModelMetricsHistory()
      self._plotTrainnedRNNModelMetricsGraphs()
      self._predictWithTrainnedModel()
      self._denormalizePredictedValues()
      self._plotActualValuesPredictedValuesGraphComparison()
      self._printActualValuesPredictedValuesRegressionMetrics()

   def plotAllRNNModelsComparisonGraph(self):
      if self.plot_all_rnn_models_comparison_graph_boolean == True:
         chosen_column_data_actual_values_to_compare_chunk_valid = self.chosen_column_data_actual_values_to_compare_chunk is not None and any(self.chosen_column_data_actual_values_to_compare_chunk)
         simplernn_predicted_values_valid = self.simplernn_predicted_values is not None and any(self.simplernn_predicted_values)
         lstm_predicted_values_valid = self.lstm_predicted_values is not None and any(self.lstm_predicted_values)
         gru_predicted_values_valid = self.gru_predicted_values is not None and any(self.gru_predicted_values)
         if chosen_column_data_actual_values_to_compare_chunk_valid and (simplernn_predicted_values_valid or lstm_predicted_values_valid or gru_predicted_values_valid):
            ## PLOT GRAPH: (X = «Days», Y = «Actual Values» Vs [«Predicted Values (SimpleRNN)» Vs «Predicted Values (LSTM)» Vs «Predicted Values (GRU)»])
            graph_title = self.cryptocoin_name + " Predictor"
            y_label = self.chosen_column
            x_label = "Days"
            x_ticks_size = len(self.chosen_column_data_actual_values_to_compare_chunk)
            curves_labels = []
            curves_colors = []
            curves_datas = []
            if chosen_column_data_actual_values_to_compare_chunk_valid:
               curves_labels.append("Actual Value")
               curves_colors.append("gold")
               curves_datas.append(self.chosen_column_data_actual_values_to_compare_chunk)
            if simplernn_predicted_values_valid:
               curves_labels.append("Predicted Value (SimpleRNN)")
               curves_colors.append("firebrick")
               curves_datas.append(self.simplernn_predicted_values)
            if lstm_predicted_values_valid:
               curves_labels.append("Predicted Value (LSTM)")
               curves_colors.append("lightseagreen")
               curves_datas.append(self.lstm_predicted_values)
            if gru_predicted_values_valid:
               curves_labels.append("Predicted Value (GRU)")
               curves_colors.append("darkslateblue")
               curves_datas.append(self.gru_predicted_values)
            self.graphPlotter.plotCurvesGraph(graph_title,
                                              y_label,
                                              x_label,
                                              x_ticks_size,
                                              curves_labels,
                                              curves_colors,
                                              curves_datas)

   def executeInvestmentSimulation(self):
      dataset_data_lines_count = self._getDatasetFileDataLinesCount()
      trainning_chunk_size = int(dataset_data_lines_count * self.trainning_data_percent)
      testing_chunk_size = dataset_data_lines_count - trainning_chunk_size
      start_index = trainning_chunk_size
      end_index = start_index + testing_chunk_size

      ## GET «Date» COLUMN ACTUAL VALUES TO COMPARE
      date_column_data = self.cryptocoin_dataset[["Date"]].values
      date_column_data_actual_values_to_compare_chunk = self.datasetManager.getActualValuesToCompareChunk(date_column_data, start_index, end_index, self.learning_size, self.prediction_size)

      ## GET «Open» COLUMN ACTUAL VALUES TO COMPARE
      open_column_data = self.cryptocoin_dataset[["Open"]].values
      open_column_data_actual_values_to_compare_chunk = self.datasetManager.getActualValuesToCompareChunk(open_column_data, start_index, end_index, self.learning_size, self.prediction_size)

      date_column_data_actual_values_to_compare_chunk_valid = date_column_data_actual_values_to_compare_chunk is not None and any(date_column_data_actual_values_to_compare_chunk)
      open_column_data_actual_values_to_compare_chunk_valid = open_column_data_actual_values_to_compare_chunk is not None and any(open_column_data_actual_values_to_compare_chunk)
      chosen_column_data_actual_values_to_compare_chunk_valid = self.chosen_column_data_actual_values_to_compare_chunk is not None and any(self.chosen_column_data_actual_values_to_compare_chunk)
      simplernn_predicted_values_valid = self.simplernn_predicted_values is not None and any(self.simplernn_predicted_values)
      lstm_predicted_values_valid = self.lstm_predicted_values is not None and any(self.lstm_predicted_values)
      gru_predicted_values_valid = self.gru_predicted_values is not None and any(self.gru_predicted_values)

      ## PRINT RNN MODELS «chosen_column»'s PREDICTED VALUES COMPARED TO ACTUAL VALUES (ROW-BY-ROW)
      for row in range(len(self.chosen_column_data_actual_values_to_compare_chunk)):
         date_column_string = ""
         open_column_string = ""
         chosen_column_string = ""
         simplernn_chosen_column_prediction_string = ""
         lstm_chosen_column_prediction_string = ""
         gru_chosen_column_prediction_string = ""
         if date_column_data_actual_values_to_compare_chunk_valid:
            date_column_string = "Date = " + str(date_column_data_actual_values_to_compare_chunk[row])
         if open_column_data_actual_values_to_compare_chunk_valid:
            open_column_string = " | Open = " + str(open_column_data_actual_values_to_compare_chunk[row])
         if chosen_column_data_actual_values_to_compare_chunk_valid:
            chosen_column_string = " | "+self.chosen_column+" = " + str(self.chosen_column_data_actual_values_to_compare_chunk[row])
         if simplernn_predicted_values_valid:
            simplernn_chosen_column_prediction_string = " | SimpleRNN_" + self.chosen_column + "_Prediction = " + str(self.simplernn_predicted_values[row])
         if lstm_predicted_values_valid:
            lstm_chosen_column_prediction_string = " | LSTM_" + self.chosen_column + "_Prediction = " + str(self.lstm_predicted_values[row])
         if gru_predicted_values_valid:
            gru_chosen_column_prediction_string = " | GRU_" + self.chosen_column + "_Prediction = " + str(self.gru_predicted_values[row])
         print(date_column_string + open_column_string + chosen_column_string + simplernn_chosen_column_prediction_string + lstm_chosen_column_prediction_string + gru_chosen_column_prediction_string)

      usd_bars_labels = []
      usd_bars_colors = []
      usd_bars_heights = []
      usd_graph_title = "NaiveInvestor Simulation"
      usd_graph_subtitle = "Initial Balance: " + str(self.initial_balance_in_usd) + " USD + " + str(self.initial_balance_in_bitcoin) + " Bitcoin" + "\nSelling Bitcoin Strategy Percent: " + str(self.selling_bitcoin_strategy_percent) + " | Buying Bitcoin Strategy Percent: " + str(self.buying_bitcoin_strategy_percent)
      usd_y_label = "USD Balance"
      usd_x_label = "Predictor"

      bitcoin_bars_labels = []
      bitcoin_bars_colors = []
      bitcoin_bars_heights = []
      bitcoin_graph_title = "NaiveInvestor Simulation"
      bitcoin_graph_subtitle = "Initial Balance: " + str(self.initial_balance_in_usd) + " USD + " + str(self.initial_balance_in_bitcoin) + " Bitcoin" + "\nSelling Bitcoin Strategy Percent: " + str(self.selling_bitcoin_strategy_percent) + " | Buying Bitcoin Strategy Percent: " + str(self.buying_bitcoin_strategy_percent)
      bitcoin_y_label = "Bitcoin Balance"
      bitcoin_x_label = "Predictor"

      if chosen_column_data_actual_values_to_compare_chunk_valid:
         ## SIMULATE INVESTIMENTS WITH ACTUAL VALUES (AS IF THEY WERE PERFECTLY PREDICTED, HYPOTHETICALLY)
         hypothetical_perfect_prediction_usd_balance = None
         hypothetical_perfect_prediction_bitcoin_balance = None
         self.naiveInvestor.setBalanceInUSD(self.initial_balance_in_usd)
         self.naiveInvestor.setBalanceInBitcoin(self.initial_balance_in_bitcoin)
         print("«Hypothetical Perfect Prediction» Initial Balance in USD: " + str(self.naiveInvestor.getBalanceInUSD()))
         print("«Hypothetical Perfect Prediction» Initial Balance in Bitcoin: " + str(self.naiveInvestor.getBalanceInBitcoin()))
         for row in range(len(self.chosen_column_data_actual_values_to_compare_chunk)):
            self.naiveInvestor.investmentAction(date_column_data_actual_values_to_compare_chunk[row],
                                                open_column_data_actual_values_to_compare_chunk[row],
                                                self.chosen_column_data_actual_values_to_compare_chunk[row])
         hypothetical_perfect_prediction_usd_balance = self.naiveInvestor.getBalanceInUSD()
         hypothetical_perfect_prediction_bitcoin_balance = self.naiveInvestor.getBalanceInBitcoin()
         print("«Hypothetical Perfect Prediction» Final Balance in USD: " + str(hypothetical_perfect_prediction_usd_balance))
         print("«Hypothetical Perfect Prediction» Final Balance in Bitcoin: " + str(hypothetical_perfect_prediction_bitcoin_balance))
         usd_bars_labels.append("Hypothetical Perfect Prediction")
         usd_bars_colors.append("gold")
         usd_bars_heights.append(float(hypothetical_perfect_prediction_usd_balance))
         bitcoin_bars_labels.append("Hypothetical Perfect Prediction")
         bitcoin_bars_colors.append("gold")
         bitcoin_bars_heights.append(float(hypothetical_perfect_prediction_bitcoin_balance))

      if simplernn_predicted_values_valid:
         ## SIMULATE INVESTIMENTS WITH «SimpleRNN» LAYER BASED RNN MODEL PREDICTED VALUES
         simplernn_usd_balance = None
         simplernn_bitcoin_balance = None
         self.naiveInvestor.setBalanceInUSD(self.initial_balance_in_usd)
         self.naiveInvestor.setBalanceInBitcoin(self.initial_balance_in_bitcoin)
         print("«SimpleRNN» Initial Balance in USD: " + str(self.naiveInvestor.getBalanceInUSD()))
         print("«SimpleRNN» Initial Balance in Bitcoin: " + str(self.naiveInvestor.getBalanceInBitcoin()))
         for row in range(len(self.chosen_column_data_actual_values_to_compare_chunk)):
            self.naiveInvestor.investmentAction(date_column_data_actual_values_to_compare_chunk[row],
                                                open_column_data_actual_values_to_compare_chunk[row],
                                                self.simplernn_predicted_values[row])
         simplernn_usd_balance = self.naiveInvestor.getBalanceInUSD()
         simplernn_bitcoin_balance = self.naiveInvestor.getBalanceInBitcoin()
         print("«SimpleRNN» Final Balance in USD: " + str(simplernn_usd_balance))
         print("«SimpleRNN» Final Balance in Bitcoin: " + str(simplernn_bitcoin_balance))
         usd_bars_labels.append("SimpleRNN")
         usd_bars_colors.append("firebrick")
         usd_bars_heights.append(float(simplernn_usd_balance))
         bitcoin_bars_labels.append("SimpleRNN")
         bitcoin_bars_colors.append("firebrick")
         bitcoin_bars_heights.append(float(simplernn_bitcoin_balance))

      if lstm_predicted_values_valid:
         ## SIMULATE INVESTIMENTS WITH «LSTM» LAYER BASED RNN MODEL PREDICTED VALUES
         lstm_usd_balance = None
         lstm_bitcoin_balance = None
         self.naiveInvestor.setBalanceInUSD(self.initial_balance_in_usd)
         self.naiveInvestor.setBalanceInBitcoin(self.initial_balance_in_bitcoin)
         print("«LSTM» Initial Balance in USD: " + str(self.naiveInvestor.getBalanceInUSD()))
         print("«LSTM» Initial Balance in Bitcoin: " + str(self.naiveInvestor.getBalanceInBitcoin()))
         for row in range(len(self.chosen_column_data_actual_values_to_compare_chunk)):
            self.naiveInvestor.investmentAction(date_column_data_actual_values_to_compare_chunk[row],
                                                open_column_data_actual_values_to_compare_chunk[row],
                                                self.lstm_predicted_values[row])
         lstm_usd_balance = self.naiveInvestor.getBalanceInUSD()
         lstm_bitcoin_balance = self.naiveInvestor.getBalanceInBitcoin()
         print("«LSTM» Final Balance in USD: " + str(lstm_usd_balance))
         print("«LSTM» Final Balance in Bitcoin: " + str(lstm_bitcoin_balance))
         usd_bars_labels.append("LSTM")
         usd_bars_colors.append("lightseagreen")
         usd_bars_heights.append(float(lstm_usd_balance))
         bitcoin_bars_labels.append("LSTM")
         bitcoin_bars_colors.append("lightseagreen")
         bitcoin_bars_heights.append(float(lstm_bitcoin_balance))

      if gru_predicted_values_valid:
         ## SIMULATE INVESTIMENTS WITH «GRU» LAYER BASED RNN MODEL PREDICTED VALUES
         gru_usd_balance = None
         gru_bitcoin_balance = None
         self.naiveInvestor.setBalanceInUSD(self.initial_balance_in_usd)
         self.naiveInvestor.setBalanceInBitcoin(self.initial_balance_in_bitcoin)
         print("«GRU» Initial Balance in USD: " + str(self.naiveInvestor.getBalanceInUSD()))
         print("«GRU» Initial Balance in Bitcoin: " + str(self.naiveInvestor.getBalanceInBitcoin()))
         for row in range(len(self.chosen_column_data_actual_values_to_compare_chunk)):
            self.naiveInvestor.investmentAction(date_column_data_actual_values_to_compare_chunk[row],
                                                open_column_data_actual_values_to_compare_chunk[row],
                                                self.gru_predicted_values[row])
         gru_usd_balance = self.naiveInvestor.getBalanceInUSD()
         gru_bitcoin_balance = self.naiveInvestor.getBalanceInBitcoin()
         print("«GRU» Final Balance in USD: " + str(gru_usd_balance))
         print("«GRU» Final Balance in Bitcoin: " + str(gru_bitcoin_balance))
         usd_bars_labels.append("GRU")
         usd_bars_colors.append("darkslateblue")
         usd_bars_heights.append(float(gru_usd_balance))
         bitcoin_bars_labels.append("GRU")
         bitcoin_bars_colors.append("darkslateblue")
         bitcoin_bars_heights.append(float(gru_bitcoin_balance))

      usd_greatest_order_of_magnitude = math.floor(math.log10(max(usd_bars_heights)))

      self.graphPlotter.plotBarsGraph(usd_graph_title,
                                      usd_graph_subtitle,
                                      usd_y_label,
                                      usd_x_label,
                                      usd_bars_labels,
                                      usd_bars_colors,
                                      usd_bars_heights,
                                      usd_greatest_order_of_magnitude)

      bitcoin_greatest_order_of_magnitude = math.floor(math.log10(max(bitcoin_bars_heights)))

      self.graphPlotter.plotBarsGraph(bitcoin_graph_title,
                                      bitcoin_graph_subtitle,
                                      bitcoin_y_label,
                                      bitcoin_x_label,
                                      bitcoin_bars_labels,
                                      bitcoin_bars_colors,
                                      bitcoin_bars_heights,
                                      bitcoin_greatest_order_of_magnitude)

def main():
   cryptoScrutator = CryptoScrutator()

   cryptoScrutator.loadCryptoScrutatorSettings("settings/crypto_scrutator_settings") ## LOAD «crypto_scrutator_settings»
   cryptoScrutator.loadDatasetSettings("settings/dataset_settings") ## LOAD «dataset_settings»
   cryptoScrutator.loadRNNModelHyperparametersSettings("settings/rnn_model_hyperparameters") ## LOAD «rnn_model_hyperparameters»
   cryptoScrutator.loadInvestmentSimulatorSettings("settings/investment_simulator_settings") ## LOAD «investment_simulator_settings»

   cryptoScrutator.handleDatasetFileChosenAndSortingColumnsNullData() ## HANDLE «dataset_file»'s «chosen_column» AND «sorting_column» NULL DATA
   cryptoScrutator.verifyDatasetFileHasEnoughDataToBePartitioned() ## VERIFY IF «dataset_file» HAS ENOUGH DATA TO BE PARTITIONED

   cryptoScrutator.loadCryptocoinDatasetCSV() ## LOAD «dataset_file» AS «cryptocoin_dataset»
   cryptoScrutator.sortCryptocoinDatasetBySortingColumn() ## SORT «cryptocoin_dataset» BY «sorting_column» (ASCENDING MODE)

   cryptoScrutator.setChosenColumnDataActualValuesToCompareChunk() ## SET «chosen_column_data_actual_values_to_compare_chunk» (FOR FUTURE COMPARISON AFTER MODEL PREDICTING PHASE)

   cryptoScrutator.normalizeChosenColumnData() ## NORMALIZE «cryptocoin_dataset»'s «chosen_column»
   cryptoScrutator.splitNormalizedChosenColumnDataBetweenTrainningAndTestingChunks() ## SPLIT «normalized_chosen_column_data» BETWEEN «normalized_trainning_data_chunk» AND «normalized_testing_data_chunk» CHUNKS
   cryptoScrutator.splitNormalizedTrainningChunkBetweenLearningAndPredictionChunks() ## SPLIT «normalized_trainning_data_chunk» BETWEEN «normalized_trainning_data_learning_chunk» AND «normalized_trainning_data_prediction_chunk» CHUNKS
   cryptoScrutator.setNormalizedTestingDataToPredictChunk() ## SET «normalized_testing_data_to_predict_chunk» FROM «normalized_testing_data_chunk»

   ## EXECUTE «SimpleRNN» LAYER BASED RNN MODEL
   cryptoScrutator.setRNNModelType("SimpleRNN")
   cryptoScrutator.executeRNNModel()

   ## EXECUTE «LSTM» LAYER BASED RNN MODEL
   cryptoScrutator.setRNNModelType("LSTM")
   cryptoScrutator.executeRNNModel()

   ## EXECUTE «GRU» LAYER BASED RNN MODEL
   cryptoScrutator.setRNNModelType("GRU")
   cryptoScrutator.executeRNNModel()

   cryptoScrutator.plotAllRNNModelsComparisonGraph() ## PLOT ALL RNN MODELS COMPARISON GRAPH

   cryptoScrutator.executeInvestmentSimulation() ## EXECUTE INVESTMENT SIMULATION

if __name__ == "__main__":
   main()
