class NaiveInvestor:

   def __init__(self):
      self.balance_in_usd = None
      self.balance_in_bitcoin = None
      self.selling_bitcoin_strategy_percent = None
      self.buying_bitcoin_strategy_percent = None

   def setBalanceInUSD(self, balance_in_usd):
      self.balance_in_usd = balance_in_usd

   def getBalanceInUSD(self):
      return self.balance_in_usd

   def setBalanceInBitcoin(self, balance_in_bitcoin):
      self.balance_in_bitcoin = balance_in_bitcoin

   def getBalanceInBitcoin(self):
      return self.balance_in_bitcoin

   def setSellingBitcoinStrategyPercent(self, selling_bitcoin_strategy_percent):
      self.selling_bitcoin_strategy_percent = selling_bitcoin_strategy_percent

   def getSellingBitcoinStrategyPercent(self):
      return self.selling_bitcoin_strategy_percent

   def setBuyingBitcoinStrategyPercent(self, buying_bitcoin_strategy_percent):
      self.buying_bitcoin_strategy_percent = buying_bitcoin_strategy_percent

   def getBuyingBitcoinStrategyPercent(self):
      return self.buying_bitcoin_strategy_percent

   def investmentAction(self, date, currency, crypto_currency, crypto_currency_open_value, crypto_currency_close_value):
      if currency == "USD" and crypto_currency == "BTC":
         if crypto_currency_open_value == crypto_currency_close_value:
            ## Cryptocurrency Stabilization (Do Nothing Strategy).
            print(str(date) + " --> " + crypto_currency + "'s Open Value: " + str(crypto_currency_open_value) + " | " + crypto_currency + "'s Close Value: " + str(crypto_currency_close_value) + " (Do Nothing...)")
         elif crypto_currency_open_value > crypto_currency_close_value:
            ## Cryptocurrency Will Devaluate (Sell Strategy).
            print(str(date) + " --> " + crypto_currency + "'s Open Value: " + str(crypto_currency_open_value) + " | " + crypto_currency + "'s Close Value: " + str(crypto_currency_close_value) + " (It is Better to Sell " + crypto_currency + "...)")
            current_balance_in_bitcoin = self.getBalanceInBitcoin()
            if current_balance_in_bitcoin > 0:
               current_balance_in_usd = self.getBalanceInUSD()
               available_bitcoin_amount_for_selling = self.getBalanceInBitcoin() * self.getSellingBitcoinStrategyPercent()
               bitcoin_to_usd_convertion = available_bitcoin_amount_for_selling * crypto_currency_open_value
               self.setBalanceInBitcoin(current_balance_in_bitcoin - available_bitcoin_amount_for_selling)
               self.setBalanceInUSD(current_balance_in_usd + bitcoin_to_usd_convertion)
         elif crypto_currency_open_value < crypto_currency_close_value:
            ## Cryptocurrency Will Appreciate (Buy Strategy).
            print(str(date) + " --> " + crypto_currency + "'s Open Value: " + str(crypto_currency_open_value) + " | " + crypto_currency + "'s Close Value: " + str(crypto_currency_close_value) + " (It is Better to Buy " + crypto_currency + "...)")
            current_balance_in_usd = self.getBalanceInUSD()
            if current_balance_in_usd > 0:
               current_balance_in_bitcoin = self.getBalanceInBitcoin()
               available_usd_amount_for_buying = self.getBalanceInUSD() * self.getBuyingBitcoinStrategyPercent()
               usd_to_bitcoin_convertion = available_usd_amount_for_buying / crypto_currency_open_value
               self.setBalanceInUSD(current_balance_in_usd - available_usd_amount_for_buying)
               self.setBalanceInBitcoin(current_balance_in_bitcoin + usd_to_bitcoin_convertion)
