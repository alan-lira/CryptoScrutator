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
         coinbase_valid_granularities_in_seconds = [60, 300, 900, 3600, 21600, 86400]
         if self.getFetchGranularityInSeconds() in coinbase_valid_granularities_in_seconds:
            url = "https://api.pro.coinbase.com/products/" + str(self.getCurrencyPair()) + "/candles?start=" + str(self.getFetchStartDate()) + "&end=" + str(self.getFetchEndDate()) + "&granularity=" + str(self.getFetchGranularityInSeconds())
            ## REMEMBER: The maximum number of data points for a single request is 300 candles.
            ##           If you wish to retrieve fine granularity data over a larger time range,
            ##           you will need to make multiple requests with new start/end ranges.
            response = requests.get(url)
            if response.status_code == 200: # HTTP 200 --> OK (Successful Request from «Coinbase» API)
               response_json_data = json.loads(response.text)
               self.setResponseJSONData(response_json_data)
            else:
               print("ERROR: COULD NOT FETCH DATA FROM «Coinbase» API!")
               raise SystemExit
         else:
               print("ERROR: «Coinbase» API ONLY ACCEPTS THE FOLLOWING GRANULARITIES (IN SECONDS): " + str(coinbase_valid_granularities_in_seconds))
               print("UPDATE «fetch_granularity_in_seconds» PARAMETER WITH ONE OF THEM.")
               raise SystemExit
      elif self.getExchanger() == "Kraken":
         kraken_valid_granularities_in_seconds = [60, 300, 900, 3600, 43200, 86400]
         if self.getFetchGranularityInSeconds() in kraken_valid_granularities_in_seconds:
            url = "https://api.kraken.com/0/public/OHLC?pair=" + str(self.getCurrencyPair().replace("-", "")) + "&interval=" + str(self.getFetchGranularityInSeconds() / 60)
            ## REMEMBER: The maximum number of data points for a single request is 720 candles.
            ##           If you wish to retrieve fine granularity data over a larger time range,
            ##           you will need to make multiple requests with new start/end ranges.
            response = requests.get(url)
            if response.status_code == 200: # HTTP 200 --> OK (Successful Request from «Kraken» API)
               response_json_data = json.loads(response.text)
               self.setResponseJSONData(response_json_data)
            else:
               print("ERROR: COULD NOT FETCH DATA FROM «Kraken» API!")
               raise SystemExit
         else:
               print("ERROR: «Kraken» API ONLY ACCEPTS THE FOLLOWING GRANULARITIES (IN SECONDS): " + str(kraken_valid_granularities_in_seconds))
               print("UPDATE «fetch_granularity_in_seconds» PARAMETER WITH ONE OF THEM.")
               raise SystemExit

   def generateDataset(self):
      dataset_directory = "datasets/"
      dataset_extension = ".csv"
      dataframe = None
      if self.getExchanger() == "Coinbase":
         dataframe = pandas.DataFrame(self.getResponseJSONData(), columns = ["Unix", "Open", "High", "Low", "Close", "Volume " + self.getCurrencyPair().split("-")[0]])
         if dataframe is not None:
            dataframe.insert(0, "Date", pandas.to_datetime(dataframe["Unix"], unit = "s"), True)
            dataframe.drop("Unix", axis = 1, inplace = True)
            fetch_start_date_formatted = dateutil.parser.parse(str(dataframe.tail(1)["Date"].tolist()[0])).strftime("%Y-%m-%d")
            fetch_end_date_formatted = dateutil.parser.parse(str(dataframe.head(1)["Date"].tolist()[0])).strftime("%Y-%m-%d")
            days_count = (dateutil.parser.parse(fetch_end_date_formatted) - dateutil.parser.parse(fetch_start_date_formatted)).days + 1
            dataset_name = self.getExchanger() + "_" + self.getCurrencyPair() + "_from_" + str(fetch_start_date_formatted) + "_to_" + str(fetch_end_date_formatted) + "(" + str(days_count) + "_days)"
            dataset_file_name = dataset_directory + dataset_name + dataset_extension
            dataframe.to_csv(dataset_file_name, index = False)
         else:
            print("ERROR: NO DATA RETRIEVED FROM «" + self.getExchanger() + "» Exchanger!")
            raise SystemExit
      elif self.getExchanger() == "Kraken":
         result = self.getResponseJSONData()["result"]
         keys = []
         for item in result:
            keys.append(item)
         dataframe = pandas.DataFrame(result[keys[0]], columns = ["Unix", "Open", "High", "Low", "Close", "VWAP", "Volume " + self.getCurrencyPair().split("-")[0], "TradeCount"])
         if dataframe is not None:
            dataframe.insert(0, "Date", pandas.to_datetime(dataframe["Unix"], unit = "s"), True)
            dataframe.drop("Unix", axis = 1, inplace = True)
            dataframe.drop("VWAP", axis = 1, inplace = True)
            dataframe.drop("TradeCount", axis = 1, inplace = True)
            dataframe.sort_values("Date")
            fetch_start_date_formatted = dateutil.parser.parse(str(dataframe.head(1)["Date"].tolist()[0])).strftime("%Y-%m-%d")
            fetch_end_date_formatted = dateutil.parser.parse(str(dataframe.tail(1)["Date"].tolist()[0])).strftime("%Y-%m-%d")
            days_count = (dateutil.parser.parse(fetch_end_date_formatted) - dateutil.parser.parse(fetch_start_date_formatted)).days + 1
            dataset_name = self.getExchanger() + "_" + self.getCurrencyPair() + "_from_" + str(fetch_start_date_formatted) + "_to_" + str(fetch_end_date_formatted) + "(" + str(days_count) + "_days)"
            dataset_file_name = dataset_directory + dataset_name + dataset_extension
            dataframe.to_csv(dataset_file_name, index = False)
         else:
            print("ERROR: NO DATA RETRIEVED FROM «" + self.getExchanger() + "» Exchanger!")
            raise SystemExit
      return dataset_file_name
