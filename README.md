import math
import time

class TradeStrategy:
    # … your __init__, refresh_remaining_units, refresh_remaining_time, etc. …

    def make_trade_decision(self, intraday_data, current_data):
        ORDER_ACTION_SELL = "sell"
        ORDER_ACTION_HOLD = "hold"


        try:
            # ---- 1) PACE: TWAP buckets based on remaining time ----
            # slice interval in seconds (default 60s)
            slice_interval = int(self.config_data.get('slice_interval', 60))
            # ensure we have a positive remaining_time
            remaining_secs = max(1, self.remaining_time)
            # how many buckets left (at least 1 so we finish)
            buckets_left = math.ceil(remaining_secs / slice_interval)
            # TWAP: how many units we “should” sell each bucket
            per_bucket = math.ceil(self.remaining_units / buckets_left)

            # ---- 2) OPPORTUNISTIC: compute a “good‐price” threshold ----
            prices = intraday_data
            mean_price = sum(prices) / len(prices)
            hi = max(prices)
            lo = min(prices)
            thresh = mean_price + (hi - lo) / 2.0

            current_price = float(current_data["current"])
            units_to_sell = 0
            action = ORDER_ACTION_HOLD

            # ---- 3) Early fill if price ≥ threshold ----
            if current_price >= thresh and self.remaining_units > 0:
                units_to_sell = min(self.remaining_units, per_bucket)
                action = ORDER_ACTION_SELL
                logger.info(
                    f"[Strategy] Opportunistic SELL {units_to_sell} @ {current_price:.2f}"
                )

            # ---- 4) Otherwise, catch‐up slice ----
            elif self.remaining_units > 0 and per_bucket > 0:
                # if we’re in the last bucket, backstop fill all
                if remaining_secs <= slice_interval:
                    units_to_sell = self.remaining_units
                    logger.info(
                        f"[Strategy] Final‐minute backstop SELL {units_to_sell}"
                    )
                else:
                    units_to_sell = per_bucket
                    logger.info(
                        f"[Strategy] Sliced SELL {units_to_sell} to keep on TWAP pace"
                    )
                action = ORDER_ACTION_SELL

            else:
                logger.info("[Strategy] HOLD — nothing to do")

            return action, units_to_sell

        except Exception as e:
            logger.exception(e)
            logger.error("[Strategy] Exception in make_trade_decision")
            return ORDER_ACTION_HOLD, 0
