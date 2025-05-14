import math
from common_util import get_logger, read_config

logger = get_logger(__name__)

from sklearn.linear_model import LinearRegression
import numpy as np

class TradeStrategy:
    """Encapsulates trading strategy logic with regression model."""

    def __init__(self):
        self.current_price = -1
        self.config_data = read_config()
        self.remaining_units = int(self.config_data.get('units_per_round', 100))
        self.remaining_time = int(self.config_data.get('order_round_timeout', 600))  # 10 minutes
        self.price_history = []
        self.last_trade_price = None
        self.profit_threshold = 0.002  # 0.2% profit threshold
        self.stop_loss = -0.001  # 0.1% stop loss
        self.min_trade_interval = 30  # Minimum seconds between trades
        self.last_trade_time = 0
        self.model = LinearRegression()
        self.training_data = []
        self.training_targets = []
        self.is_trained = False

    def refresh_remaining_units(self, sold_units):
        self.remaining_units -= sold_units

    def refresh_remaining_time(self, passed_seconds):
        self.remaining_time = int(int(self.config_data.get('order_round_timeout', 300)) - passed_seconds)
        logger.info(f"Remaining Time: {self.remaining_time} seconds")

    def update_training_data(self, features, target):
        self.training_data.append(features)
        self.training_targets.append(target)
        if len(self.training_data) > 50 and not self.is_trained:
            self.train_model()

    def train_model(self):
        X = np.array(self.training_data)
        y = np.array(self.training_targets)
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Regression model trained with collected data")

    def predict_price_change(self, features):
        if not self.is_trained:
            return 0.0
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        return prediction

    def make_trade_decision(self, intraday_data, current_data):
        ORDER_ACTION_SELL = "sell"
        ORDER_ACTION_HOLD = "hold"

        try:
            current_price = float(current_data["current"])
            current_time = current_data.get("time", 0)
            self.price_history.append(current_price)
            
            # Keep only last 30 price points
            if len(self.price_history) > 30:
                self.price_history = self.price_history[-30:]

            logger.info(f"Remaining Units: {self.remaining_units}, Time: {self.remaining_time}s")

            # Prepare features for regression: e.g., last 5 price changes
            if len(self.price_history) < 6:
                features = [0]*5
            else:
                features = [ (self.price_history[-i] - self.price_history[-i-1])/self.price_history[-i-1] for i in range(1,6)]

            # Update training data with last known price change (if available)
            if len(self.price_history) > 6:
                target = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
                self.update_training_data(features, target)

            prediction = self.predict_price_change(features)
            logger.info(f"Predicted price change: {prediction}")

            # Initialize last_trade_price if not set
            if self.last_trade_price is None:
                self.last_trade_price = current_price
                return ORDER_ACTION_HOLD, 0

            # Decision based on regression prediction and thresholds
            if prediction > self.profit_threshold and self.remaining_units > 0:
                trade_size = min(math.ceil(self.remaining_units * 0.5), self.remaining_units)
                self.last_trade_price = current_price
                self.last_trade_time = current_time
                logger.info("[Trade Strategy] Regression model suggests SELL")
                return ORDER_ACTION_SELL, trade_size
            elif prediction < self.stop_loss:
                trade_size = min(math.ceil(self.remaining_units * 0.5), self.remaining_units)
                self.last_trade_price = current_price
                self.last_trade_time = current_time
                logger.info("[Trade Strategy] Regression model suggests STOP LOSS SELL")
                return ORDER_ACTION_SELL, trade_size
            else:
                logger.info("[Trade Strategy] Hold position")
                return ORDER_ACTION_HOLD, 0

        except Exception as e:
            logger.exception(e)
            logger.error(f"[Trade Strategy] Exception found")

        return ORDER_ACTION_HOLD, 0
