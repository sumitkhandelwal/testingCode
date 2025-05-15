import math
import numpy as pd
import numpy as np
from common_util import get_logger, read_config
from stock_prediction_v2 import StockPredictor

logger = get_logger(__name__)

class TradeStrategy:
    """Encapsulates trading strategy logic."""

    def __init__(self):
        self.current_price = -1
        self.config_data = read_config()
        self.remaining_units = int(self.config_data.get('units_per_round', 100))
        self.remaining_time = int(self.config_data.get('order_round_timeout', 300))
        self.predictor = StockPredictor()
        self.predictor.load_trained_model()
        self.price_history = []
        self.volume_history = []

    def refresh_remaining_units(self, sold_units):
        self.remaining_units -= sold_units

    def refresh_remaining_time(self, passed_seconds):
        self.remaining_time = int(self.config_data.get('order_round_timeout', 300)) - passed_seconds
        logger.info(f"Remaining Time: {self.remaining_time} seconds")

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands."""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, sma, lower_band

    def calculate_macd(self, prices):
        """Calculate MACD for recent prices."""
        if len(prices) < 26:
            return 0, 0
            
        ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = ema12 - ema26
        signal = pd.Series([macd]).ewm(span=9).mean().iloc[-1]
        return macd, signal

    def calculate_units_to_sell(self, trend_strength, time_pressure, volatility):
        """Calculate dynamic number of units to sell based on market conditions."""
        base_units = math.ceil(self.remaining_units * 0.2)  # Start with 20% base
        
        # Adjust based on trend strength (0 to 1)
        trend_adjustment = 1 + trend_strength
        
        # Adjust based on time pressure (increases as time runs out)
        time_factor = 1 + (1 - (self.remaining_time / 600))  # 600 seconds = 10 minutes
        
        # Adjust based on volatility (0 to 1)
        volatility_adjustment = 1 + volatility
        
        # Calculate final units
        units = min(
            self.remaining_units,
            math.ceil(base_units * trend_adjustment * time_factor * volatility_adjustment)
        )
        
        # Ensure minimum units are sold
        return max(1, min(units, self.remaining_units))

    def make_trade_decision(self, intraday_data, current_data):
        ORDER_ACTION_SELL = "sell"
        ORDER_ACTION_HOLD = "hold"

        try:
            logger.info(f"Remaining Units: {self.remaining_units}")
            
            if not intraday_data:
                return ORDER_ACTION_HOLD, 0

            current_price = float(current_data["current"])
            self.price_history.append(current_price)
            self.volume_history.append(float(current_data.get("turnover", 0)))

            # Keep only recent history
            max_history = 100
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
                self.volume_history = self.volume_history[-max_history:]

            # Calculate technical indicators
            rsi = self.calculate_rsi(self.price_history) if len(self.price_history) > 14 else 50
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(self.price_history)
            macd, macd_signal = self.calculate_macd(self.price_history)

            # Prepare data for prediction
            if len(self.price_history) >= 2:
                recent_prices = pd.DataFrame({
                    'current': self.price_history,
                    'turnover': self.volume_history,
                    'high': self.price_history,  # Simplified for demonstration
                    'low': self.price_history,   # Simplified for demonstration
                    'change': np.diff(self.price_history + [current_price]),
                    'percent': np.diff(self.price_history + [current_price]) / self.price_history[-1] * 100
                })
                
                # Get prediction if enough data is available
                try:
                    X, _ = self.predictor.prepare_features(recent_prices)
                    prediction = self.predictor.predict(X)[-1]  # Get latest prediction
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    prediction = 0

                # Calculate trend strength (0 to 1)
                trend_strength = abs(macd) / (max(self.price_history) - min(self.price_history))
                
                # Calculate volatility (0 to 1)
                volatility = (upper_band - lower_band) / middle_band

                # Decision making logic
                if self.remaining_units <= 0:
                    logger.info("[Trade Strategy] All units sold out")
                    return ORDER_ACTION_HOLD, 0
                
                # Emergency time-based selling
                if self.remaining_time <= 20:
                    logger.info("[Trade Strategy] Emergency time-based selling")
                    return ORDER_ACTION_SELL, self.remaining_units

                # Strong sell signals
                if (rsi > 70 and current_price > upper_band) or \
                   (prediction < -0.5 and macd < macd_signal):
                    units = self.calculate_units_to_sell(trend_strength, 0.8, volatility)
                    logger.info("[Trade Strategy] Strong sell signal")
                    return ORDER_ACTION_SELL, units

                # Moderate sell signals
                if (rsi > 60 and current_price > middle_band) or \
                   (prediction < -0.2 and current_price > middle_band):
                    units = self.calculate_units_to_sell(trend_strength, 0.5, volatility)
                    logger.info("[Trade Strategy] Moderate sell signal")
                    return ORDER_ACTION_SELL, units

                # Weak sell signals
                if self.remaining_time < 300 and self.remaining_units > 50:  # Last 5 minutes with >50% units
                    units = self.calculate_units_to_sell(trend_strength, 0.3, volatility)
                    logger.info("[Trade Strategy] Time-based sell signal")
                    return ORDER_ACTION_SELL, units

            logger.info("[Trade Strategy] No suitable strategy")
            return ORDER_ACTION_HOLD, 0

        except Exception as e:
            logger.exception(e)
            logger.error("[Trade Strategy] Exception found")
            return ORDER_ACTION_HOLD, 0
