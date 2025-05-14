import math
from common_util import get_logger, read_config

logger = get_logger(__name__)

class TradeStrategy:
    """Encapsulates trading strategy logic."""

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

    def refresh_remaining_units(self, sold_units):
        self.remaining_units -= sold_units

    def refresh_remaining_time(self, passed_seconds):
        self.remaining_time = int(int(self.config_data.get('order_round_timeout', 300)) - passed_seconds)
        logger.info(f"Remaining Time: {self.remaining_time} seconds")

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
            
            # Initialize last_trade_price if not set
            if self.last_trade_price is None:
                self.last_trade_price = current_price
                return ORDER_ACTION_HOLD, 0

            # Calculate technical indicators
            mean_price = sum(self.price_history) / len(self.price_history)
            highest_price = max(self.price_history[-10:])  # Last 10 points
            lowest_price = min(self.price_history[-10:])  # Last 10 points
            
            # Calculate price momentum
            price_change = (current_price - self.price_history[-2]) / self.price_history[-2] if len(self.price_history) > 1 else 0
            
            # Calculate RSI-like indicator
            gains = sum(1 for i in range(1, len(self.price_history)) if self.price_history[i] > self.price_history[i-1])
            total_periods = len(self.price_history) - 1
            rsi = (gains / total_periods * 100) if total_periods > 0 else 50

            # Time-based urgency factor (0 to 1)
            urgency = 1 - (self.remaining_time / 600)
            
            # Determine optimal trade size based on remaining time and units
            base_trade_size = math.ceil(self.remaining_units * 0.25)  # Start with 25% of remaining
            time_adjusted_size = math.ceil(base_trade_size * (1 + urgency))
            
            # Decision making logic
            should_sell = False
            trade_size = 0

            # Emergency sell with 20 seconds remaining
            if self.remaining_time <= 20 and self.remaining_units > 0:
                logger.info("[Trade Strategy] Emergency time-based sell")
                return ORDER_ACTION_SELL, self.remaining_units

            # Check if enough time has passed since last trade
            if current_time - self.last_trade_time < self.min_trade_interval:
                return ORDER_ACTION_HOLD, 0

            # Strong uptrend detection
            if current_price > highest_price * 1.001 and self.remaining_units > 0:
                should_sell = True
                trade_size = time_adjusted_size
                logger.info("[Trade Strategy] Strong uptrend detected")

            # Profit taking
            elif current_price > self.last_trade_price * (1 + self.profit_threshold) and self.remaining_units > 0:
                should_sell = True
                trade_size = time_adjusted_size
                logger.info("[Trade Strategy] Profit threshold reached")

            # RSI-based overbought condition
            elif rsi > 70 and self.remaining_units > 0:
                should_sell = True
                trade_size = base_trade_size
                logger.info("[Trade Strategy] Overbought condition")

            # Stop loss trigger
            elif current_price < self.last_trade_price * (1 + self.stop_loss) and self.remaining_units > 0:
                should_sell = True
                trade_size = math.ceil(self.remaining_units * 0.5)  # Sell half on stop loss
                logger.info("[Trade Strategy] Stop loss triggered")

            # Time-based gradual selling
            elif self.remaining_time < 300 and self.remaining_units > 40:  # Last 5 minutes with >40% remaining
                should_sell = True
                trade_size = math.ceil(self.remaining_units * (urgency * 0.4))  # Gradually increase size
                logger.info("[Trade Strategy] Time-based gradual sell")

            if should_sell and trade_size > 0:
                trade_size = min(trade_size, self.remaining_units)  # Don't sell more than we have
                self.last_trade_price = current_price
                self.last_trade_time = current_time
                return ORDER_ACTION_SELL, trade_size

            return ORDER_ACTION_HOLD, 0

        except Exception as e:
            logger.exception(e)
            logger.error(f"[Trade Strategy] Exception found")

        return ORDER_ACTION_HOLD, 0
