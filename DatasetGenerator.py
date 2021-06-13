import dateutil.parser
import json
import pandas
import requests

class DatasetGenerator:

   def __init__(self):
      self.exchanger = None
      self.currency_pair = None
      self.fetch_granularity_in_seconds = None
      self.fetch_start_date = None
      self.fetch_end_date = None
      self.response_json_data = None

   def setExchanger(self, exchanger):
      self.exchanger = exchanger

   def getExchanger(self):
      return self.exchanger

   def setCurrencyPair(self, currency_pair):
      self.currency_pair = currency_pair

   def getCurrencyPair(self):
      return self.currency_pair

   def setFetchGranularityInSeconds(self, fetch_granularity_in_seconds):
      self.fetch_granularity_in_seconds = fetch_granularity_in_seconds

   def getFetchGranularityInSeconds(self):
      return self.fetch_granularity_in_seconds

   def setFetchStartDate(self, fetch_start_date):
      self.fetch_start_date = fetch_start_date

   def getFetchStartDate(self):
      return self.fetch_start_date

   def setFetchEndDate(self, fetch_end_date):
      self.fetch_end_date = fetch_end_date

   def getFetchEndDate(self):
      return self.fetch_end_date

   def setResponseJSONData(self, response_json_data):
      self.response_json_data = response_json_data

   def getResponseJSONData(self):
      return self.response_json_data

   def fetchJSONData(self):
      if self.getExchanger() == "Coinbase":
         url = "https://api.pro.coinbase.com/products/" + str(self.getCurrencyPair()) + "/candles?start=" + str(self.getFetchStartDate()) + "&end=" + str(self.getFetchEndDate()) + "&granularity=" + str(self.getFetchGranularityInSeconds())
         response = requests.get(url)
         if response.status_code == 200: # HTTP 200 --> OK (Successful Request from Coinbase API)
            response_json_data = json.loads(response.text)
            self.setResponseJSONData(response_json_data)
         else:
            print("ERROR: COULD NOT FETCH DATA FROM «Coinbase» API!")
            raise SystemExit

   def generateDataset(self):
      days_count = (dateutil.parser.parse(self.getFetchEndDate()) - dateutil.parser.parse(self.getFetchStartDate())).days
      fetch_start_date_formatted = dateutil.parser.parse(self.getFetchStartDate()).strftime("%Y-%m-%d")
      fetch_end_date_formatted = dateutil.parser.parse(self.getFetchEndDate()).strftime("%Y-%m-%d")
      dataframe = None
      dataset_directory = "datasets/"
      dataset_name = self.getExchanger() + "_" + self.getCurrencyPair() + "_from_" + str(fetch_start_date_formatted) + "_to_" + str(fetch_end_date_formatted) + "(" + str(days_count) + "_days)"
      dataset_extension = ".csv"
      dataset_file_name = dataset_directory + dataset_name + dataset_extension
      if self.getExchanger() == "Coinbase":
         dataframe = pandas.DataFrame(self.getResponseJSONData(), columns = ["Unix", "Open", "High", "Low", "Close", "Volume " + self.getCurrencyPair().split("-")[0]])
         if dataframe is not None:
            dataframe.insert(0, "Date", pandas.to_datetime(dataframe["Unix"], unit = "s"), True)
            dataframe.drop("Unix", axis = 1, inplace = True)
            dataframe.to_csv(dataset_file_name, index = False)
         else:
            print("ERROR: NO DATA RETRIEVED FROM «" + self.getExchanger() + "» Exchanger!")
            raise SystemExit
      return dataset_file_name
