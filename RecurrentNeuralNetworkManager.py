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
      self.prediction_history = []

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

   def addDenseLayer(self, number_of_dense_units):
      dense_layer = Dense(units = number_of_dense_units)
      self.model.add(dense_layer)

   def summarizeModel(self):
      print(self.model.summary())

   def trainModel(self, x_train, y_train, validation_split_percent, number_of_epochs, batch_size, shuffle_boolean):
      trainned_model_result = self.model.fit(x_train,
                                             y_train,
                                             validation_split = validation_split_percent,
                                             epochs = number_of_epochs,
                                             batch_size = batch_size,
                                             shuffle = shuffle_boolean,
                                             verbose = 2)
      self.trainned_model_metrics_history = trainned_model_result.history

   def getTrainnedModelMetricsHistory(self):
      return self.trainned_model_metrics_history

   def printTrainnedModelMetricsHistory(self):
      trainned_model_metrics_history = self.getTrainnedModelMetricsHistory()
      for metric_name in trainned_model_metrics_history:
         print("Metric '"+str(metric_name)+"':")
         print(str(trainned_model_metrics_history[metric_name])+"\n")

   def predictWithTrainnedModel(self, x_test):
      prediction_result = self.model.predict(x_test)
      self.prediction_history = prediction_result

   def getPredictionHistory(self):
      return self.prediction_history
