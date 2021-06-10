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
      self.plot_loss_graph_boolean = None
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
      self.simplernn_predicted_prices = None
      self.lstm_predicted_prices = None
      self.gru_predicted_prices = None

   def loadCryptoScrutatorSettings(self, crypto_scrutator_settings_file):
      with open(crypto_scrutator_settings_file, 'r') as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "print_metrics_history_boolean":
                  self.print_metrics_history_boolean = value == "True"
               elif key == "plot_loss_graph_boolean":
                  self.plot_loss_graph_boolean = value == "True"
               elif key == "plot_real_price_vs_predicted_price_graph_boolean":
                  self.plot_real_price_vs_predicted_price_graph_boolean = value == "True"
               elif key == "print_regression_metrics_boolean":
                  self.print_regression_metrics_boolean = value == "True"
               elif key == "plot_all_rnn_models_comparison_graph_boolean":
                  self.plot_all_rnn_models_comparison_graph_boolean = value == "True"

   def loadDatasetSettings(self, dataset_settings_file):
      with open(dataset_settings_file, 'r') as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "cryptocoin_name":
                  self.cryptocoin_name = str(value)
               elif key == "dataset_file":
                  self.dataset_file = str(value)
               elif key == "sorting_column":
                  self.sorting_column = str(value)
               elif key == "chosen_column":
                  self.chosen_column = str(value)

   def loadRNNModelHyperparametersSettings(self, rnn_model_hyperparameters_settings_file):
      with open(rnn_model_hyperparameters_settings_file, 'r') as settings_file:
         for line in settings_file:
            line = line.strip()
            splitted_line = line.split(" = ")
            key = splitted_line[0]
            value = splitted_line[1]
            if len(splitted_line) > 1:
               if key == "model_name":
                  self.model_name = str(value)
               elif key == "trainning_data_percent":
                  self.trainning_data_percent = float(value)
               elif key == "learning_size":
                  self.learning_size = int(value)
               elif key == "prediction_size":
                  self.prediction_size = int(value)
               elif key == "number_of_simplernn_units":
                  self.number_of_simplernn_units = int(value)
               elif key == "number_of_lstm_units":
                  self.number_of_lstm_units = int(value)
               elif key == "number_of_gru_units":
                  self.number_of_gru_units = int(value)
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
                     self.number_of_time_steps = int(value)
               elif key == "number_of_features":
                  self.number_of_features = int(value)
               elif key == "input_shape":
                  if value == "(number_of_time_steps, number_of_features)":
                     self.input_shape = (self.number_of_time_steps, self.number_of_features)
                  else:
                     self.input_shape = value
               elif key == "negative_slope_coefficient":
                  self.negative_slope_coefficient = float(value)
               elif key == "fraction_of_the_input_units_to_drop":
                  self.fraction_of_the_input_units_to_drop = float(value)
               elif key == "number_of_dense_units":
                  self.number_of_dense_units = int(value)
               elif key == "loss_function_name":
                  self.loss_function_name = str(value)
               elif key == "optimizer_name":
                  self.optimizer_name = str(value)
               elif key == "optimizer_learning_rate":
                  self.optimizer_learning_rate = float(value)
               elif key == "metrics_list":
                  self.metrics_list = value.split(", ")
               elif key == "validation_split_percent":
                  self.validation_split_percent = float(value)
               elif key == "number_of_epochs":
                  self.number_of_epochs = int(value)
               elif key == "batch_size":
                  if value == "learning_size":
                     self.batch_size = self.learning_size
                  else:
                     self.batch_size = int(value)
               elif key == "shuffle_boolean":
                  self.shuffle_boolean = value == "True"

   def loadCryptocoinDatasetCSV(self):
      self.cryptocoin_dataset = self.datasetManager.loadDataset(self.dataset_file)

   def sortCryptocoinDatasetByColumn(self):
      self.cryptocoin_dataset = self.datasetManager.sortDatasetByColumn(self.cryptocoin_dataset, self.sorting_column)

   def handleChosenColumnNullData(self):
      #print("'" + self.chosen_column + "' Column has null values: " + str(self.datasetManager.checkIfDatasetColumnHasNullValues(self.cryptocoin_dataset, self.chosen_column)))
      #print("'" + self.chosen_column + "' Column null values' count: " + str(self.datasetManager.datasetColumnNullValuesCount(self.cryptocoin_dataset, self.chosen_column)))
      pass

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

   def executeRNNModel(self):
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

      ## COMPILE MODEL
      self.recurrentNeuralNetworkManager.compileSequentialModel()

      ## SUMMARIZE MODEL
      self.recurrentNeuralNetworkManager.summarizeModel()

      ## TRAIN MODEL
      self.recurrentNeuralNetworkManager.trainModel(self.normalized_trainning_data_learning_chunk,
                                                    self.normalized_trainning_data_prediction_chunk,
                                                    self.validation_split_percent,
                                                    self.number_of_epochs,
                                                    self.batch_size,
                                                    self.shuffle_boolean)

      if self.print_metrics_history_boolean == True:
         ## PRINT TRAINNED MODEL METRICS HISTORY
         self.recurrentNeuralNetworkManager.printTrainnedModelMetricsHistory()

      if self.plot_loss_graph_boolean == True:
         ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Loss' Vs 'Validation Loss')
         trainned_model_metrics_history = self.recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
         graph_title = "Training and Validation Loss ("+self.rnn_model_type+")"
         y_label = "Loss"
         first_curve_label = "Training Loss"
         first_curve_color = "blue"
         first_curve_data = trainned_model_metrics_history["loss"]
         second_curve_label = "Validation Loss"
         second_curve_color = "orange"
         second_curve_data = trainned_model_metrics_history["val_loss"]
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

      ## PREDICT WITH TRAINNED MODEL
      self.recurrentNeuralNetworkManager.predictWithTrainnedModel(self.normalized_testing_data_to_predict_chunk)
      prediction_history = self.recurrentNeuralNetworkManager.getPredictionHistory()

      ## REVERT NORMALIZING STEP (INVERSE TRANSFORM)
      predicted_prices = self.datasetManager.inverseTransformData(prediction_history)

      if self.rnn_model_type == "SimpleRNN":
         self.simplernn_predicted_prices = predicted_prices
      elif self.rnn_model_type == "LSTM":
         self.lstm_predicted_prices = predicted_prices
      elif self.rnn_model_type == "GRU":
         self.gru_predicted_prices = predicted_prices

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

      ## CLEAR MODEL
      self.recurrentNeuralNetworkManager.clearModel()

      if self.print_regression_metrics_boolean == True:
         ## PRINT REGRESSION METRICS
         if self.rnn_model_type == "SimpleRNN":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.simplernn_predicted_prices)
         elif self.rnn_model_type == "LSTM":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.lstm_predicted_prices)
         elif self.rnn_model_type == "GRU":
            self.datasetManager.printRegressionMetrics(self.testing_data_real_values_chunk, self.gru_predicted_prices)

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

cryptoScrutator = CryptoScrutator()
cryptoScrutator.loadCryptoScrutatorSettings("settings/crypto_scrutator_settings") ## LOAD CRYPTO SCRUTATOR SETTINGS
cryptoScrutator.loadDatasetSettings("settings/dataset_settings") ## LOAD DATASET SETTINGS
cryptoScrutator.loadRNNModelHyperparametersSettings("settings/rnn_model_hyperparameters") ## LOAD RNN MODEL HYPERPARAMETERS SETTINGS
cryptoScrutator.loadCryptocoinDatasetCSV() ## LOAD CRYPTOCOIN DATASET
cryptoScrutator.sortCryptocoinDatasetByColumn() ## SORT CRYPTOCOIN DATASET BY 'Date' COLUMN (ASCENDING MODE)
cryptoScrutator.handleChosenColumnNullData() ## HANDLE CHOSEN COLUMN's ('Close') NULL DATA
cryptoScrutator.normalizeChosenColumnData() ## NORMALIZE CRYPTOCOIN DATASET's 'Close' COLUMN (CLOSE PRICE)
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
