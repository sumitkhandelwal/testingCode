import math
from common_util import get_logger, read_config
from stock_prediction import StockPredictor
import numpy as np

logger = get_logger(__name__)

class TradeStrategy:
    """
    Encapsulates trading strategy logic based on intraday price data, momentum, and remaining units/time.
    Integrates stock price change prediction model for optimized selling decisions.

    Attributes:
        current_price (float): Current stock price.
        config_data (dict): Configuration data loaded from config file.
        remaining_units (int): Number of units left to sell.
        remaining_time (int): Time left for order round timeout in seconds.
        price_threshold (float): Threshold for significant price movements (default 2%).
        last_sell_price (float or None): Price at which last sell occurred.
        attempts_left (int): Number of selling attempts left.
        units_per_window (int): Units to sell per time window.
        stock_predictor (StockPredictor): Model for price change prediction.

    Methods:
        refresh_remaining_units(sold_units):
            Updates remaining units after selling.

        refresh_remaining_time(passed_seconds):
            Updates remaining time left for trading.

        predict_price_change(intraday_data):
            Predicts price change using the stock prediction model.

        make_trade_decision(intraday_data, current_data):
            Makes trade decision (sell or hold) based on predicted price change and constraints.
    """

    def __init__(self):
        self.current_price = -1
        self.config_data = read_config()
        self.units_per_window = int(self.config_data.get('units_per_round', 100))
        self.remaining_units = self.units_per_window
        self.remaining_time = int(self.config_data.get('order_round_timeout', 600))  # 600 seconds as per user
        self.price_threshold = 0.02  # 2% threshold for significant price movements
        self.last_sell_price = None
        self.attempts_left = 3  # User constraint: only 3 attempts
        self.stock_predictor = StockPredictor()
        self.stock_predictor.load_trained_model()

    def refresh_remaining_units(self, sold_units):
        """
        Update the remaining units after selling.

        Args:
            sold_units (int): Number of units sold.
        """
        self.remaining_units -= sold_units
        if self.remaining_units < 0:
            self.remaining_units = 0

    def refresh_remaining_time(self, passed_seconds):
        """
        Update the remaining time left for trading.

        Args:
            passed_seconds (int): Seconds passed since start of order round.
        """
        self.remaining_time = int(int(self.config_data.get('order_round_timeout', 600)) - passed_seconds)
        logger.info(f"Remaining Time: {self.remaining_time} seconds")

    def predict_price_change(self, intraday_data):
        """
        Predict price change using the stock prediction model.

        Args:
            intraday_data (list or array): List of recent prices.

        Returns:
            float: Predicted price change.
        """
        try:
            # Prepare data for prediction: create feature array with required shape
            # Use last sequence_length prices repeated for features except 'current'
            # Since we only have price, create dummy features for other columns as zeros
            seq_len = self.stock_predictor.sequence_length
            if len(intraday_data) < seq_len:
                logger.info("Not enough data for prediction")
                return 0.0

            # Create dummy features array: shape (seq_len, feature_count)
            feature_count = len(self.stock_predictor.feature_cols)
            data = np.zeros((seq_len, feature_count))
            # Fill 'current' price column index with intraday_data last seq_len prices normalized
            current_idx = self.stock_predictor.feature_cols.index('current')
            prices = np.array(intraday_data[-seq_len:])
            # Normalize prices between 0 and 1 for model input
            min_p, max_p = prices.min(), prices.max()
            if max_p - min_p == 0:
                norm_prices = np.zeros_like(prices)
            else:
                norm_prices = (prices - min_p) / (max_p - min_p)
            data[:, current_idx] = norm_prices

            X = np.array([data])
            pred_change = self.stock_predictor.predict(X)[0]
            logger.info(f"Predicted price change: {pred_change}")
            return pred_change
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

    def make_trade_decision(self, intraday_data, current_data):
        """
        Make trade decision (sell or hold) based on predicted price change and constraints.

        Args:
            intraday_data (list or array): List of recent prices.
            current_data (dict): Current price data with key "current".

        Returns:
            tuple: (action, units)
                action (str): "sell" or "hold"
                units (int): Number of units to sell if action is "sell", else 0
        """
        ORDER_ACTION_SELL = "sell"
        ORDER_ACTION_HOLD = "hold"

        try:
            logger.info(f"Remaining Units: {self.remaining_units}, Attempts Left: {self.attempts_left}")

            if self.remaining_units <= 0 or self.attempts_left <= 0:
                logger.info("[Trade Strategy] No units or attempts left")
                return ORDER_ACTION_HOLD, 0

            if len(intraday_data) < self.stock_predictor.sequence_length:
                logger.info("[Trade Strategy] Not enough data for prediction, holding")
                return ORDER_ACTION_HOLD, 0

            current_price = float(current_data["current"])

            # Predict price change
            predicted_change = self.predict_price_change(intraday_data)

            # Decision logic:
            # Sell units_per_window if predicted change is positive and attempts left
            # Otherwise hold
            # If time running out (< 20s), sell all remaining units regardless

            if self.remaining_time < 20 and self.remaining_units > 0:
                logger.info("[Trade Strategy] Time running out, emergency sell")
                units_to_sell = self.remaining_units
                self.refresh_remaining_units(units_to_sell)
                self.attempts_left -= 1
                return ORDER_ACTION_SELL, units_to_sell

            if predicted_change > 0 and self.attempts_left > 0:
                units_to_sell = min(self.units_per_window, self.remaining_units)
                logger.info(f"[Trade Strategy] Selling {units_to_sell} units based on positive predicted change")
                self.refresh_remaining_units(units_to_sell)
                self.attempts_left -= 1
                return ORDER_ACTION_SELL, units_to_sell

            logger.info("[Trade Strategy] Holding for better conditions")
            return ORDER_ACTION_HOLD, 0

        except Exception as e:
            logger.exception(e)
            logger.error("[Trade Strategy] Exception found")
            return ORDER_ACTION_HOLD, 0

##########################


import unittest
from tradestrategy import TradeStrategy

class TestTradeStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TradeStrategy()
        # Mock config values for consistent testing
        self.strategy.units_per_window = 100
        self.strategy.remaining_units = 100
        self.strategy.remaining_time = 600
        self.strategy.attempts_left = 3

    def test_initial_state(self):
        self.assertEqual(self.strategy.remaining_units, 100)
        self.assertEqual(self.strategy.remaining_time, 600)
        self.assertEqual(self.strategy.attempts_left, 3)

    def test_refresh_remaining_units(self):
        self.strategy.refresh_remaining_units(30)
        self.assertEqual(self.strategy.remaining_units, 70)
        self.strategy.refresh_remaining_units(80)
        self.assertEqual(self.strategy.remaining_units, 0)  # Should not go negative

    def test_refresh_remaining_time(self):
        self.strategy.refresh_remaining_time(100)
        self.assertEqual(self.strategy.remaining_time, 500)

    def test_predict_price_change_not_enough_data(self):
        # Provide less data than sequence length
        short_data = [100] * (self.strategy.stock_predictor.sequence_length - 1)
        pred = self.strategy.predict_price_change(short_data)
        self.assertEqual(pred, 0.0)

    def test_make_trade_decision_hold_due_to_no_attempts(self):
        self.strategy.attempts_left = 0
        intraday_data = [100] * self.strategy.stock_predictor.sequence_length
        current_data = {"current": 100}
        action, units = self.strategy.make_trade_decision(intraday_data, current_data)
        self.assertEqual(action, "hold")
        self.assertEqual(units, 0)

    def test_make_trade_decision_emergency_sell(self):
        self.strategy.remaining_time = 10
        self.strategy.remaining_units = 50
        self.strategy.attempts_left = 2
        intraday_data = [100] * self.strategy.stock_predictor.sequence_length
        current_data = {"current": 100}
        action, units = self.strategy.make_trade_decision(intraday_data, current_data)
        self.assertEqual(action, "sell")
        self.assertEqual(units, 50)
        self.assertEqual(self.strategy.remaining_units, 0)
        self.assertEqual(self.strategy.attempts_left, 1)

    def test_make_trade_decision_sell_on_positive_prediction(self):
        # Patch predict_price_change to return positive value
        self.strategy.predict_price_change = lambda x: 0.05
        self.strategy.remaining_units = 100
        self.strategy.attempts_left = 3
        intraday_data = [100] * self.strategy.stock_predictor.sequence_length
        current_data = {"current": 100}
        action, units = self.strategy.make_trade_decision(intraday_data, current_data)
        self.assertEqual(action, "sell")
        self.assertEqual(units, 100)
        self.assertEqual(self.strategy.remaining_units, 0)
        self.assertEqual(self.strategy.attempts_left, 2)

    def test_make_trade_decision_hold_on_negative_prediction(self):
        # Patch predict_price_change to return negative value
        self.strategy.predict_price_change = lambda x: -0.05
        self.strategy.remaining_units = 100
        self.strategy.attempts_left = 3
        intraday_data = [100] * self.strategy.stock_predictor.sequence_length
        current_data = {"current": 100}
        action, units = self.strategy.make_trade_decision(intraday_data, current_data)
        self.assertEqual(action, "hold")
        self.assertEqual(units, 0)
        self.assertEqual(self.strategy.remaining_units, 100)
        self.assertEqual(self.strategy.attempts_left, 3)

if __name__ == "__main__":
    unittest.main()

