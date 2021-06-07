from keras.layers import Dense, Dropout, LeakyReLU, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

class RecurrentNeuralNetworkManager:

   def __init__(self):
      self.model = None
      self.loss_function = None
      self.optimizer = None
      self.metrics = []
      self.trainned_model_metrics_history = []

   def createEmptySequentialModel(self, model_name):
      self.model = Sequential(name = model_name)

   def loadMeanSquaredErrorLossFunction(self):
      self.loss_function = "mean_squared_error"

   def loadAdamOptimizer(self, learning_rate):
      self.optimizer = Adam(learning_rate = learning_rate)

   def loadMetrics(self, metrics_list):
      self.metrics = metrics_list

   def compileSequentialModel(self):
      self.model.compile(loss = self.loss_function, optimizer = self.optimizer, metrics = self.metrics)

   def addLongShortTermMemoryLayer(self, number_of_lstm_units, activation_function, input_shape):
      lstm_layer = LSTM(units = number_of_lstm_units, activation = activation_function, input_shape = input_shape)
      self.model.add(lstm_layer)

   def addLeakyRectifiedLinearUnitLayer(self, negative_slope_coefficient):
      leaky_relu_layer = LeakyReLU(alpha = negative_slope_coefficient)
      self.model.add(leaky_relu_layer)

   def addDropoutLayer(self, fraction_of_the_input_units_to_drop):
      dropout_layer = Dropout(fraction_of_the_input_units_to_drop)
      self.model.add(dropout_layer)

   def addDenseLayer(self, output_space_dimensionality):
      dense_layer = Dense(units = output_space_dimensionality)
      self.model.add(dense_layer)

   def summarizeModel(self):
      print(self.model.summary())

   def trainModel(self, X, Y, validation_split_percent, number_of_epochs, batch_size, shuffle_boolean):
      trainned_model_result = self.model.fit(X,
                                             Y,
                                             validation_split = validation_split_percent,
                                             epochs = number_of_epochs,
                                             batch_size = batch_size,
                                             shuffle = shuffle_boolean,
                                             verbose = 2)
      self.trainned_model_metrics_history = trainned_model_result.history

   def printTrainnedModelMetricsHistory(self):
      for metric_name in self.trainned_model_metrics_history:
         print("Metric '"+str(metric_name)+"':")
         print(str(self.trainned_model_metrics_history[metric_name])+"\n")
