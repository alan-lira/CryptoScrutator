import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # AVOID TENSORFLOW LOGGING
from DatasetManager import *
from GraphPlotter import *
from RecurrentNeuralNetworkManager import *

def executeLSTMModel(model_name,
                     number_of_lstm_units,
                     activation,
                     recurrent_activation,
                     input_shape,
                     negative_slope_coefficient,
                     fraction_of_the_input_units_to_drop,
                     number_of_dense_units,
                     loss_function_name,
                     optimizer_name,
                     optimizer_learning_rate,
                     metrics_list,
                     normalized_past_train_data_chunk,
                     normalized_future_train_data_chunk,
                     validation_split_percent,
                     number_of_epochs,
                     batch_size,
                     shuffle_boolean,
                     real_prices):
   ## CREATE LSTM MODEL
   recurrentNeuralNetworkManager.createEmptySequentialModel(model_name)

   ## ADD LSTM LAYER
   recurrentNeuralNetworkManager.addLongShortTermMemoryLayer(number_of_lstm_units,
                                                             activation,
                                                             recurrent_activation,
                                                             input_shape)

   ## ADD LEAKYRELU LAYER
   recurrentNeuralNetworkManager.addLeakyRectifiedLinearUnitLayer(negative_slope_coefficient)

   ## ADD DROPOUT LAYER
   recurrentNeuralNetworkManager.addDropoutLayer(fraction_of_the_input_units_to_drop)

   ## ADD DENSE LAYER
   recurrentNeuralNetworkManager.addDenseLayer(number_of_dense_units)

   ## LOAD LOSS FUNCTION
   recurrentNeuralNetworkManager.loadLossFunction(loss_function_name)

   ## LOAD OPTIMIZER
   recurrentNeuralNetworkManager.loadOptimizer(optimizer_name, optimizer_learning_rate)

   ## LOAD METRICS
   recurrentNeuralNetworkManager.loadMetrics(metrics_list)

   ## COMPILE MODEL
   recurrentNeuralNetworkManager.compileSequentialModel()

   ## SUMMARIZE MODEL
   recurrentNeuralNetworkManager.summarizeModel()

   ## TRAIN MODEL
   recurrentNeuralNetworkManager.trainModel(normalized_past_train_data_chunk,
                                            normalized_future_train_data_chunk,
                                            validation_split_percent,
                                            number_of_epochs,
                                            batch_size,
                                            shuffle_boolean)

   ## PRINT TRAINNED MODEL METRICS HISTORY
   recurrentNeuralNetworkManager.printTrainnedModelMetricsHistory()

   ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Loss' Vs 'Validation Loss')
   lstm_trainned_model_metrics_history = recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
   graph_title = "Training and Validation Loss (LSTM)"
   y_label = "Loss"
   first_curve_label = "Training Loss"
   first_curve_color = "blue"
   first_curve_data = lstm_trainned_model_metrics_history["loss"]
   second_curve_label = "Validation Loss"
   second_curve_color = "orange"
   second_curve_data = lstm_trainned_model_metrics_history["val_loss"]
   x_label = "Number of Epochs"
   x_ticks_size = number_of_epochs
   graphPlotter.plotTwoCurvesGraph(graph_title,
                                   y_label,
                                   first_curve_label,
                                   first_curve_color,
                                   first_curve_data,
                                   second_curve_label,
                                   second_curve_color,
                                   second_curve_data,
                                   x_label,
                                   x_ticks_size)

   ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Accuracy' Vs 'Validation Accuracy')
   lstm_trainned_model_metrics_history = recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
   graph_title = "Training and Validation Accuracy (LSTM)"
   y_label = "Accuracy"
   first_curve_label = "Training Accuracy"
   first_curve_color = "blue"
   first_curve_data = lstm_trainned_model_metrics_history["accuracy"]
   second_curve_label = "Validation Accuracy"
   second_curve_color = "orange"
   second_curve_data = lstm_trainned_model_metrics_history["val_accuracy"]
   x_label = "Number of Epochs"
   x_ticks_size = number_of_epochs
   graphPlotter.plotTwoCurvesGraph(graph_title,
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
   recurrentNeuralNetworkManager.predictWithTrainnedModel(normalized_past_test_data_chunk)
   lstm_prediction_history = recurrentNeuralNetworkManager.getPredictionHistory()

   ## REVERT NORMALIZING STEP (INVERSE TRANSFORM)
   lstm_predicted_prices = datasetManager.inverseTransformData(lstm_prediction_history)

   ## PLOT GRAPH: (X = 'Days', Y = 'Real Price' Vs 'Predicted Price')
   graph_title = cryptocoin + " Price Predictor"
   y_label = "Close Price (USD)"
   first_curve_label = "Real Price"
   first_curve_color = "blue"
   first_curve_data = real_prices
   second_curve_label = "Predicted Price (LSTM)"
   second_curve_color = "orange"
   second_curve_data = lstm_predicted_prices
   x_label = "Days"
   x_ticks_size = len(real_prices)
   graphPlotter.plotTwoCurvesGraph(graph_title,
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
   recurrentNeuralNetworkManager.clearModel()

   ## PRINT REGRESSION METRICS
   datasetManager.printRegressionMetrics(real_prices, lstm_predicted_prices)

   return lstm_predicted_prices

def executeGRUModel(model_name,
                    number_of_gru_units,
                    activation,
                    recurrent_activation,
                    input_shape,
                    negative_slope_coefficient,
                    fraction_of_the_input_units_to_drop,
                    number_of_dense_units,
                    loss_function_name,
                    optimizer_name,
                    optimizer_learning_rate,
                    metrics_list,
                    normalized_past_train_data_chunk,
                    normalized_future_train_data_chunk,
                    validation_split_percent,
                    number_of_epochs,
                    batch_size,
                    shuffle_boolean,
                    real_prices):
   ## CREATE GRU MODEL
   recurrentNeuralNetworkManager.createEmptySequentialModel(model_name)

   ## ADD GRU LAYER
   recurrentNeuralNetworkManager.addGatedRecurrentUnitLayer(number_of_gru_units,
                                                            activation,
                                                            recurrent_activation,
                                                            input_shape)

   ## ADD LEAKYRELU LAYER
   recurrentNeuralNetworkManager.addLeakyRectifiedLinearUnitLayer(negative_slope_coefficient)

   ## ADD DROPOUT LAYER
   recurrentNeuralNetworkManager.addDropoutLayer(fraction_of_the_input_units_to_drop)

   ## ADD DENSE LAYER
   recurrentNeuralNetworkManager.addDenseLayer(number_of_dense_units)

   ## LOAD LOSS FUNCTION
   recurrentNeuralNetworkManager.loadLossFunction(loss_function_name)

   ## LOAD OPTIMIZER
   recurrentNeuralNetworkManager.loadOptimizer(optimizer_name, optimizer_learning_rate)

   ## LOAD METRICS
   recurrentNeuralNetworkManager.loadMetrics(metrics_list)

   ## COMPILE MODEL
   recurrentNeuralNetworkManager.compileSequentialModel()

   ## SUMMARIZE MODEL
   recurrentNeuralNetworkManager.summarizeModel()

   ## TRAIN MODEL
   recurrentNeuralNetworkManager.trainModel(normalized_past_train_data_chunk,
                                            normalized_future_train_data_chunk,
                                            validation_split_percent,
                                            number_of_epochs,
                                            batch_size,
                                            shuffle_boolean)

   ## PRINT TRAINNED MODEL METRICS HISTORY
   recurrentNeuralNetworkManager.printTrainnedModelMetricsHistory()

   ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Loss' Vs 'Validation Loss')
   gru_trainned_model_metrics_history = recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
   graph_title = "Training and Validation Loss (GRU)"
   y_label = "Loss"
   first_curve_label = "Training Loss"
   first_curve_color = "blue"
   first_curve_data = gru_trainned_model_metrics_history["loss"]
   second_curve_label = "Validation Loss"
   second_curve_color = "orange"
   second_curve_data = gru_trainned_model_metrics_history["val_loss"]
   x_label = "Number of Epochs"
   x_ticks_size = number_of_epochs
   graphPlotter.plotTwoCurvesGraph(graph_title,
                                   y_label,
                                   first_curve_label,
                                   first_curve_color,
                                   first_curve_data,
                                   second_curve_label,
                                   second_curve_color,
                                   second_curve_data,
                                   x_label,
                                   x_ticks_size)

   ## PLOT GRAPH: (X = 'Number of Epochs', Y = 'Training Accuracy' Vs 'Validation Accuracy')
   gru_trainned_model_metrics_history = recurrentNeuralNetworkManager.getTrainnedModelMetricsHistory()
   graph_title = "Training and Validation Accuracy (GRU)"
   y_label = "Accuracy"
   first_curve_label = "Training Accuracy"
   first_curve_color = "blue"
   first_curve_data = gru_trainned_model_metrics_history["accuracy"]
   second_curve_label = "Validation Accuracy"
   second_curve_color = "orange"
   second_curve_data = gru_trainned_model_metrics_history["val_accuracy"]
   x_label = "Number of Epochs"
   x_ticks_size = number_of_epochs
   graphPlotter.plotTwoCurvesGraph(graph_title,
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
   recurrentNeuralNetworkManager.predictWithTrainnedModel(normalized_past_test_data_chunk)
   gru_prediction_history = recurrentNeuralNetworkManager.getPredictionHistory()

   ## REVERT NORMALIZING STEP (INVERSE TRANSFORM)
   gru_predicted_prices = datasetManager.inverseTransformData(gru_prediction_history)

   ## PLOT GRAPH: (X = 'Days', Y = 'Real Price' Vs 'Predicted Price')
   graph_title = cryptocoin + " Price Predictor"
   y_label = "Close Price (USD)"
   first_curve_label = "Real Price"
   first_curve_color = "blue"
   first_curve_data = real_prices
   second_curve_label = "Predicted Price (GRU)"
   second_curve_color = "orange"
   second_curve_data = gru_predicted_prices
   x_label = "Days"
   x_ticks_size = len(real_prices)
   graphPlotter.plotTwoCurvesGraph(graph_title,
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
   recurrentNeuralNetworkManager.clearModel()

   ## PRINT REGRESSION METRICS
   datasetManager.printRegressionMetrics(real_prices, gru_predicted_prices)

   return gru_predicted_prices


datasetManager = DatasetManager()
graphPlotter = GraphPlotter()
recurrentNeuralNetworkManager = RecurrentNeuralNetworkManager()

cryptocoin = "Bitcoin"

## LOAD CRYPTOCOIN DATASET
cryptocoin_csv_file_path = "bitcoin_test_sample.csv" # Dataset Structure: [Date | Symbol | Open | High | Low | Close | Volume BTC | Volume USD]
dataset = datasetManager.loadDataset(cryptocoin_csv_file_path)

## SORT CRYPTOCOIN DATASET BY 'Date' COLUMN (ASCENDING MODE)
date_column = "Date"
dataset = datasetManager.sortDatasetByColumn(dataset, date_column)

## CHOSEN COLUMN OF DATASET: 'Close' (CLOSE PRICE)
chosen_column = "Close"
print("Column has null values? " + str(datasetManager.checkIfDatasetColumnHasNullValues(dataset, chosen_column)))
print("Column null values count: " + str(datasetManager.datasetColumnNullValuesCount(dataset, chosen_column)))

## PRINT CRYPTOCOIN DATASET's FIRST ROWS (HEAD)
number_of_rows_to_print = 10
datasetManager.printDataset(dataset, number_of_rows_to_print)

## PLOT GRAPH: (X = 'Date', Y = 'Close')
graph_title = "Historical Price of " + cryptocoin
y_label = "Close Price (USD)"
y_data = dataset[[chosen_column]]
x_label = "Date"
x_data = dataset[date_column]
x_ticks_size = 30
x_ticks_rotation = 90
graphPlotter.plotOneCurveGraph(graph_title,
                               dataset,
                               y_label,
                               y_data,
                               x_label,
                               x_data,
                               x_ticks_size,
                               x_ticks_rotation)

## PRINT CRYPTOCOIN DATASET's 'Close' COLUMN (CLOSE PRICE)
datasetManager.printDatasetValuesOfColumn(dataset, chosen_column)

## NORMALIZE CRYPTOCOIN DATASET's 'Close' COLUMN (CLOSE PRICE)
normalized_chosen_column = datasetManager.normalizeDatasetValuesOfColumn(dataset, chosen_column)

## PRINT ORIGINAL AND NORMALIZED VALUES OF 'Close' COLUMN (CLOSE PRICE)
datasetManager.printOriginalAndNormalizedDatasetColumn(dataset, chosen_column, normalized_chosen_column)

## SPLIT NORMALIZED TRAIN AND TEST DATA CHUNKS (TRAINNING DATA PERCENT: 80%)
trainning_data_percent = 0.8
normalized_train_data_chunk, normalized_test_data_chunk = datasetManager.splitNormalizedTrainAndTestDataChunks(normalized_chosen_column, trainning_data_percent)

## SET AMOUNT OF DAYS TO LEARN TO PREDICT THE FUTURE TIME SERIES
past_size = 5 # LEARN 5 DAYS (DAYS OF TRAINNING)
future_size = 0 # PREDICT THE NEXT DAY

## SPLIT NORMALIZED PAST AND FUTURE TRAIN DATA CHUNKS
train_start_index = 0
train_end_index = len(normalized_train_data_chunk)
normalized_past_train_data_chunk, normalized_future_train_data_chunk = datasetManager.splitNormalizedPastAndFutureDataChunks(normalized_train_data_chunk, train_start_index, train_end_index, past_size, future_size)
print(normalized_past_train_data_chunk.shape)
print(normalized_future_train_data_chunk.shape)

## SPLIT NORMALIZED PAST AND FUTURE TEST DATA CHUNKS
test_start_index = 0
test_end_index = len(normalized_test_data_chunk)
normalized_past_test_data_chunk, normalized_future_test_data_chunk = datasetManager.splitNormalizedPastAndFutureDataChunks(normalized_test_data_chunk, test_start_index, test_end_index, past_size, future_size)

## REVERT NORMALIZING STEP (INVERSE TRANSFORM)
real_prices = datasetManager.inverseTransformData(normalized_future_test_data_chunk)

## DEFINE RNN MODEL PARAMETERS
model_name = "Crypto_Predictor"
number_of_lstm_units = 64 # LSTM NETWORK INPUT LAYER TUPLE: (number_of_samples, number_of_time_steps, number_of_features)
number_of_gru_units = 64 # GRU NETWORK INPUT LAYER TUPLE: (number_of_samples, number_of_time_steps, number_of_features)
activation = "tanh"
recurrent_activation = "sigmoid"
number_of_time_steps = past_size # COULD BE "None" [DISCOVER 'number_of_time_steps' AT TRAINNING STAGE]
number_of_features = 1 # 'Close' COLUMN (CLOSE PRICE)
input_shape = (number_of_time_steps, number_of_features) # 'number_of_samples' OMITTED [AUTOMATICALLY ASSUMES 1 OR MORE SAMPLES (1..n)]
negative_slope_coefficient = 0.5
fraction_of_the_input_units_to_drop = 0.1
number_of_dense_units = 1
loss_function_name = "mean_squared_error"
optimizer_name = "Adam"
optimizer_learning_rate = 0.0001
metrics_list = ["accuracy"]
validation_split_percent = 0.1
number_of_epochs = 50
batch_size = past_size
shuffle_boolean = False

## EXECUTE LSTM MODEL
lstm_predicted_prices = executeLSTMModel(model_name,
                                         number_of_lstm_units,
                                         activation,
                                         recurrent_activation,
                                         input_shape,
                                         negative_slope_coefficient,
                                         fraction_of_the_input_units_to_drop,
                                         number_of_dense_units,
                                         loss_function_name,
                                         optimizer_name,
                                         optimizer_learning_rate,
                                         metrics_list,
                                         normalized_past_train_data_chunk,
                                         normalized_future_train_data_chunk,
                                         validation_split_percent,
                                         number_of_epochs,
                                         batch_size,
                                         shuffle_boolean,
                                         real_prices)

## EXECUTE GRU MODEL
gru_predicted_prices = executeGRUModel(model_name,
                                       number_of_gru_units,
                                       activation,
                                       recurrent_activation,
                                       input_shape,
                                       negative_slope_coefficient,
                                       fraction_of_the_input_units_to_drop,
                                       number_of_dense_units,
                                       loss_function_name,
                                       optimizer_name,
                                       optimizer_learning_rate,
                                       metrics_list,
                                       normalized_past_train_data_chunk,
                                       normalized_future_train_data_chunk,
                                       validation_split_percent,
                                       number_of_epochs,
                                       batch_size,
                                       shuffle_boolean,
                                       real_prices)

## PLOT GRAPH: (X = 'Days', Y = 'Real Price' Vs 'Predicted Price (LSTM)' Vs 'Predicted Price (GRU)')
graph_title = cryptocoin + " Price Predictor"
y_label = "Close Price (USD)"
first_curve_label = "Real Price"
first_curve_color = "blue"
first_curve_data = real_prices
second_curve_label = "Predicted Price (LSTM)"
second_curve_color = "red"
second_curve_data = lstm_predicted_prices
third_curve_label = "Predicted Price (GRU)"
third_curve_color = "orange"
third_curve_data = gru_predicted_prices
x_label = "Days"
x_ticks_size = len(real_prices)
graphPlotter.plotThreeCurvesGraph(graph_title,
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
                                  x_label,
                                  x_ticks_size)
