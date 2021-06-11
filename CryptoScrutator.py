import csv
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # AVOID TENSORFLOW LOGGING
import string
from DatasetManager import *
from GraphPlotter import *
from RecurrentNeuralNetworkManager import *

class CryptoScrutator:

   def __init__(self):
      self.datasetManager = DatasetManager()
      self.graphPlotter = GraphPlotter()
      self.recurrentNeuralNetworkManager = RecurrentNeuralNetworkManager()
      self.print_metrics_history_boolean = None
      self.plot_metrics_graphs_boolean = None
      self.plot_real_price_vs_predicted_price_graph_boolean = None
      self.print_regression_metrics_boolean = None
      self.plot_all_rnn_models_comparison_graph_boolean = None
      self.cryptocoin_name = None
      self.dataset_file = None
      self.sorting_column = None
      self.chosen_column = None
      self.cryptocoin_dataset = None
      self.normalized_chosen_column_data = None
      self.trainning_data_percent = None
      self.normalized_trainning_data_chunk = None
      self.normalized_trainning_data_learning_chunk = None
      self.normalized_trainning_data_prediction_chunk = None
      self.normalized_testing_data_chunk = None
      self.normalized_testing_data_to_predict_chunk = None
      self.normalized_testing_data_real_values_chunk = None
      self.testing_data_real_values_chunk = None
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
      self.simplernn_predicted_prices = None
      self.lstm_predicted_prices = None
      self.gru_predicted_prices = None

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
               elif key == "plot_real_price_vs_predicted_price_graph_boolean":
                  self.plot_real_price_vs_predicted_price_graph_boolean = value == "True"
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

   def loadCryptocoinDatasetCSV(self):
      self.cryptocoin_dataset = self.datasetManager.loadDataset(self.dataset_file)

   def sortCryptocoinDatasetByColumn(self):
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
      self.normalized_chosen_column_data = self.datasetManager.normalizeDatasetValuesOfColumn(self.cryptocoin_dataset, self.chosen_column)

   def splitNormalizedChosenColumnDataBetweenTrainningAndTestingChunks(self):
      self.normalized_trainning_data_chunk, self.normalized_testing_data_chunk = self.datasetManager.splitNormalizedTrainAndTestDataChunks(self.normalized_chosen_column_data, self.trainning_data_percent)

   def splitNormalizedTrainningDataChunkBetweenLearningAndPredictionChunks(self):
      train_start_index = 0
      train_end_index = len(self.normalized_trainning_data_chunk)
      self.normalized_trainning_data_learning_chunk, self.normalized_trainning_data_prediction_chunk = self.datasetManager.splitNormalizedPastAndFutureDataChunks(self.normalized_trainning_data_chunk, train_start_index, train_end_index, self.learning_size, self.prediction_size)

   def splitNormalizedTestingDataChunkBetweenToPredictAndRealValuesChunks(self):
      test_start_index = 0
      test_end_index = len(self.normalized_testing_data_chunk)
      self.normalized_testing_data_to_predict_chunk, self.normalized_testing_data_real_values_chunk = self.datasetManager.splitNormalizedPastAndFutureDataChunks(self.normalized_testing_data_chunk, test_start_index, test_end_index, self.learning_size, self.prediction_size)

   def revertNormalizingStepForNormalizedTestingDataRealValuesChunk(self):
      self.testing_data_real_values_chunk = self.datasetManager.inverseTransformData(self.normalized_testing_data_real_values_chunk)

   def setRNNModelType(self, rnn_model_type):
      self.rnn_model_type = rnn_model_type

   def _createRNNModel(self):
      ## CREATE 'rnn_model_type' RNN MODEL
      self.recurrentNeuralNetworkManager.createEmptySequentialModel(self.model_name+"_"+self.rnn_model_type)

      if self.rnn_model_type == "SimpleRNN":
         ## ADD SIMPLERNN LAYER
         self.recurrentNeuralNetworkManager.addSimpleRecurrentNeuralNetworkLayer(self.number_of_simplernn_units,
                                                                                 self.activation,
                                                                                 self.recurrent_initializer,
                                                                                 self.input_shape)   
      elif self.rnn_model_type == "LSTM":
         ## ADD LSTM LAYER
         self.recurrentNeuralNetworkManager.addLongShortTermMemoryLayer(self.number_of_lstm_units,
                                                                        self.activation,
                                                                        self.recurrent_activation,
                                                                        self.input_shape)
      elif self.rnn_model_type == "GRU":
         ## ADD GRU LAYER
         self.recurrentNeuralNetworkManager.addGatedRecurrentUnitLayer(self.number_of_gru_units,
                                                                       self.activation,
                                                                       self.recurrent_activation,
                                                                       self.input_shape)

      ## ADD LEAKYRELU LAYER
      self.recurrentNeuralNetworkManager.addLeakyRectifiedLinearUnitLayer(self.negative_slope_coefficient)

      ## ADD DROPOUT LAYER
      self.recurrentNeuralNetworkManager.addDropoutLayer(self.fraction_of_the_input_units_to_drop)

      ## ADD DENSE LAYER
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
         ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Metric' Vs 'Validation Metric')
         trainned_model_metrics_history = self.recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
         metrics_to_plot = ["loss"]
         metrics_to_plot.extend(self.metrics_list)
         for metric in metrics_to_plot:
            metric_name = metric.replace("_", " ").title()
            training_metric = trainned_model_metrics_history[metric]
            validation_metric = trainned_model_metrics_history["val_"+metric]
            graph_title = metric_name + " Graph (" + self.rnn_model_type + ")"
            y_label = metric_name
            first_curve_label = "Training " + metric_name
            first_curve_color = "red"
            first_curve_data = training_metric
            second_curve_label = "Validation " + metric_name
            second_curve_color = "green"
            second_curve_data = validation_metric
            x_label = "Number of Epochs"
            x_ticks_size = self.number_of_epochs
            self.graphPlotter.plotTwoCurvesGraph(graph_title,
                                                 y_label,
                                                 first_curve_label,
                                                 first_curve_color,
                                                 first_curve_data,
                                                 second_curve_label,
                                                 second_curve_color,
                                                 second_curve_data,
                                                 x_label,
                                                 x_ticks_size)

   def _predictWithTrainnedModel(self):
      ## PREDICT WITH TRAINNED RNN MODEL
      self.recurrentNeuralNetworkManager.predictWithTrainnedModel(self.normalized_testing_data_to_predict_chunk)
      self.prediction_history = self.recurrentNeuralNetworkManager.getPredictionHistory()

   def _reverseNormalizingDataStep(self):
      ## REVERT NORMALIZING DATA STEP (INVERSE TRANSFORM)
      predicted_prices = self.datasetManager.inverseTransformData(self.prediction_history)
      if self.rnn_model_type == "SimpleRNN":
         self.simplernn_predicted_prices = predicted_prices
      elif self.rnn_model_type == "LSTM":
         self.lstm_predicted_prices = predicted_prices
      elif self.rnn_model_type == "GRU":
         self.gru_predicted_prices = predicted_prices

   def _plotRealPricePredictedPriceGraphComparison(self):
      ## PLOT REAL PRICE VS PREDICTED PRICE GRAPH
      if self.plot_real_price_vs_predicted_price_graph_boolean == True:
         ## PLOT GRAPH: (X = 'Days', Y = 'Real Price' Vs 'Predicted Price')
         graph_title = self.cryptocoin_name + " Price Predictor"
         y_label = self.chosen_column + " Price (USD)"
         first_curve_label = "Real Price"
         first_curve_color = "blue"
         first_curve_data = self.testing_data_real_values_chunk
         second_curve_label = "Predicted Price ("+self.rnn_model_type+")"
         second_curve_color = None
         second_curve_data = None
         if self.rnn_model_type == "SimpleRNN":
            second_curve_color = "red"
            second_curve_data = self.simplernn_predicted_prices
         elif self.rnn_model_type == "LSTM":
            second_curve_color = "orange"
            second_curve_data = self.lstm_predicted_prices
         elif self.rnn_model_type == "GRU":
            second_curve_color = "green"
            second_curve_data = self.gru_predicted_prices      
         x_label = "Days"
         x_ticks_size = len(self.testing_data_real_values_chunk)
         self.graphPlotter.plotTwoCurvesGraph(graph_title,
                                              y_label,
                                              first_curve_label,
                                              first_curve_color,
                                              first_curve_data,
                                              second_curve_label,
                                              second_curve_color,
                                              second_curve_data,
                                              x_label,
                                              x_ticks_size)

   def _clearRNNModel(self):
      ## CLEAR RNN MODEL
      self.recurrentNeuralNetworkManager.clearModel()

   def _printRealPricePredictedPriceRegressionMetrics(self):
      if self.print_regression_metrics_boolean == True:
         ## PRINT REGRESSION METRICS
         if self.rnn_model_type == "SimpleRNN":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.simplernn_predicted_prices)
         elif self.rnn_model_type == "LSTM":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.lstm_predicted_prices)
         elif self.rnn_model_type == "GRU":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.gru_predicted_prices)

   def executeRNNModel(self):
      self._createRNNModel()
      self._trainRNNModel()
      self._printTrainnedRNNModelMetricsHistory()
      self._plotTrainnedRNNModelMetricsGraphs()
      self._predictWithTrainnedModel()
      self._reverseNormalizingDataStep()
      self._plotRealPricePredictedPriceGraphComparison()
      self._printRealPricePredictedPriceRegressionMetrics()

   def plotAllRNNModelsComparisonGraph(self):
      if self.plot_all_rnn_models_comparison_graph_boolean == True:
         if any(self.testing_data_real_values_chunk) and any(self.simplernn_predicted_prices) and any(self.lstm_predicted_prices) and any(self.gru_predicted_prices):
            ## PLOT GRAPH: (X = 'Days', Y = 'Real Price' Vs 'Predicted Price (SimpleRNN)' Vs 'Predicted Price (LSTM)' Vs 'Predicted Price (GRU)')
            graph_title = self.cryptocoin_name + " Price Predictor"
            y_label = self.chosen_column + " Price (USD)"
            first_curve_label = "Real Price"
            first_curve_color = "blue"
            first_curve_data = self.testing_data_real_values_chunk
            second_curve_label = "Predicted Price (SimpleRNN)"
            second_curve_color = "red"
            second_curve_data = self.simplernn_predicted_prices
            third_curve_label = "Predicted Price (LSTM)"
            third_curve_color = "orange"
            third_curve_data = self.lstm_predicted_prices
            fourth_curve_label = "Predicted Price (GRU)"
            fourth_curve_color = "green"
            fourth_curve_data = self.gru_predicted_prices
            x_label = "Days"
            x_ticks_size = len(self.testing_data_real_values_chunk)
            self.graphPlotter.plotFourCurvesGraph(graph_title,
                                                  y_label,
                                                  first_curve_label,
                                                  first_curve_color,
                                                  first_curve_data,
                                                  second_curve_label,
                                                  second_curve_color,
                                                  second_curve_data,
                                                  third_curve_label,
                                                  third_curve_color,
                                                  third_curve_data,
                                                  fourth_curve_label,
                                                  fourth_curve_color,
                                                  fourth_curve_data,
                                                  x_label,
                                                  x_ticks_size)

def main():
   cryptoScrutator = CryptoScrutator()
   cryptoScrutator.loadCryptoScrutatorSettings("settings/crypto_scrutator_settings") ## LOAD CRYPTO SCRUTATOR SETTINGS
   cryptoScrutator.loadDatasetSettings("settings/dataset_settings") ## LOAD DATASET SETTINGS
   cryptoScrutator.loadRNNModelHyperparametersSettings("settings/rnn_model_hyperparameters") ## LOAD RNN MODEL HYPERPARAMETERS SETTINGS
   cryptoScrutator.handleDatasetFileChosenAndSortingColumnsNullData() ## HANDLE 'chosen_column' AND 'sorting_column's NULL DATA
   cryptoScrutator.verifyDatasetFileHasEnoughDataToBePartitioned() ## VERIFY IF CRYPTOCOIN's DATASET HAS ENOUGH DATA TO BE PARTITIONED
   cryptoScrutator.loadCryptocoinDatasetCSV() ## LOAD CRYPTOCOIN's DATASET
   cryptoScrutator.sortCryptocoinDatasetByColumn() ## SORT CRYPTOCOIN DATASET BY 'sorting_column' (ASCENDING MODE)
   cryptoScrutator.normalizeChosenColumnData() ## NORMALIZE CRYPTOCOIN DATASET's 'chosen_column'
   cryptoScrutator.splitNormalizedChosenColumnDataBetweenTrainningAndTestingChunks() ## SPLIT NORMALIZED CHOSEN COLUMN's DATA BETWEEN TRAINNING AND TESTING CHUNKS
   cryptoScrutator.splitNormalizedTrainningDataChunkBetweenLearningAndPredictionChunks() ## SPLIT NORMALIZED TRAINNING DATA's CHUNK BETWEEN LEARNING AND PREDICTION CHUNKS
   cryptoScrutator.splitNormalizedTestingDataChunkBetweenToPredictAndRealValuesChunks() ## SPLIT NORMALIZED TESTING DATA's CHUNK BETWEEN TO PREDICT AND REAL VALUES CHUNKS
   cryptoScrutator.revertNormalizingStepForNormalizedTestingDataRealValuesChunk() ## REVERT NORMALIZING STEP FOR NORMALIZED TESTING DATA's REAL VALUES CHUNK (INVERSE TRANSFORM)
   cryptoScrutator.setRNNModelType("SimpleRNN") ## EXECUTE SIMPLERNN LAYER BASED MODEL
   cryptoScrutator.executeRNNModel()
   cryptoScrutator.setRNNModelType("LSTM") ## EXECUTE LSTM LAYER BASED MODEL
   cryptoScrutator.executeRNNModel()
   cryptoScrutator.setRNNModelType("GRU") ## EXECUTE GRU LAYER BASED MODEL
   cryptoScrutator.executeRNNModel()
   cryptoScrutator.plotAllRNNModelsComparisonGraph() ## PLOT ALL RNN MODELS COMPARISON GRAPH

if __name__ == "__main__":
   main()
