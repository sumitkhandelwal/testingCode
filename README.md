import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_path="stock_model.h5", sequence_length=60):
        """
        Initialize the Stock Predictor with model parameters.
        
        Args:
            model_path (str): Path to save/load the model
            sequence_length (int): Number of time steps to look back for prediction
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_cols = ['current', 'turnover', 'high', 'low', 'change', 'percent']
        self.target_col = 'change'  # Changed to predict 'change' for trading decisions
        
    def load_data(self, filepath):
        """
        Load and prepare stock data from CSV.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Processed dataframe
        """
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.sort_values('datetime')
            df = df.reset_index(drop=True)
            logger.info(f"Loaded data from {filepath} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df, train=True):
        """
        Preprocess data for model training or prediction.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            train (bool): Whether in training mode (fit scaler) or prediction mode
            
        Returns:
            tuple: Processed features and targets (if train=True)
        """
        data = df[self.feature_cols].values
        
        if train:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.transform(data)
            
        if train:
            X, y = self.create_sequences(data_scaled)
            logger.info(f"Created sequences with shapes X: {X.shape}, y: {y.shape}")
            return X, y
        else:
            X = self.create_sequences_predict(data_scaled)
            logger.info(f"Created prediction sequences with shape X: {X.shape}")
            return X
            
    def create_sequences(self, data_scaled):
        """
        Create sequences for training.
        
        Returns:
            tuple: numpy arrays of features and targets
        """
        X, y = [], []
        for i in range(self.sequence_length, len(data_scaled)):
            X.append(data_scaled[i-self.sequence_length:i])
            y.append(data_scaled[i, self.feature_cols.index(self.target_col)])
        return np.array(X), np.array(y)
        
    def create_sequences_predict(self, data_scaled):
        """
        Create sequences for prediction (last sequence).
        
        Returns:
            numpy.array: Feature array for prediction
        """
        if len(data_scaled) < self.sequence_length:
            raise ValueError("Not enough data for sequence length")
        return np.array([data_scaled[-self.sequence_length:]])
        
    def build_model(self):
        """
        Build and compile the enhanced LSTM/GRU model.
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        self.model = Sequential([
            Bidirectional(GRU(128, return_sequences=True), input_shape=(self.sequence_length, len(self.feature_cols))),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(GRU(64)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logger.info("Enhanced model built and compiled")
        return self.model
        
    def train(self, X_train, y_train, validation_split=0.1, epochs=100, batch_size=32):
        """
        Train the model with early stopping and learning rate reduction.
        
        Args:
            X_train (numpy.array): Training features
            y_train (numpy.array): Training targets
            validation_split (float): Fraction of data for validation
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
        """
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        # Recompile model with gradient clipping to avoid NaNs
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        # Check for NaNs or inf in training data and replace with zeros
        import numpy as np
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            X_train = np.nan_to_num(X_train)
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            y_train = np.nan_to_num(y_train)

        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        logger.info("Training completed")
        return history
        
    def load_trained_model(self):
        """
        Load a pre-trained model if available.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            logger.info(f"Loaded pre-trained model from {self.model_path}")
        else:
            logger.warning("No pre-trained model found, building new model")
            self.build_model()
            
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X (numpy.array): Input features
            
        Returns:
            numpy.array: Predicted values
        """
        if self.model is None:
            self.load_trained_model()
            
        pred_scaled = self.model.predict(X)
        # Create dummy array for inverse transform
        pred_full = np.zeros((len(pred_scaled), len(self.feature_cols)))
        pred_full[:, self.feature_cols.index(self.target_col)] = pred_scaled[:, 0]
        pred = self.scaler.inverse_transform(pred_full)[:, self.feature_cols.index(self.target_col)]
        logger.info(f"Generated predictions with shape {pred.shape}")
        return pred
        
    def evaluate(self, X_test, y_test, df_test=None):
        """
        Evaluate model performance.
        
        Args:
            X_test (numpy.array): Test features
            y_test (numpy.array): Test targets
            df_test (pandas.DataFrame): Original test dataframe for profit calculation
            
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X_test)
        # Inverse transform actual values
        y_test_full = np.zeros((len(y_test), len(self.feature_cols)))
        y_test_full[:, self.feature_cols.index(self.target_col)] = y_test
        y_test_inv = self.scaler.inverse_transform(y_test_full)[:, self.feature_cols.index(self.target_col)]
        
        mae = mean_absolute_error(y_test_inv, y_pred)
        mse = mean_squared_error(y_test_inv, y_pred)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        direction_correct = np.mean(np.sign(y_test_inv[1:] - y_test_inv[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        
        # Improved trading simulation for profit (sell less stock but maximize profit)
        profit_metrics = {}
        if df_test is not None and len(df_test) == len(y_pred) + self.sequence_length:
            df_test = df_test.iloc[self.sequence_length:].reset_index(drop=True)
            initial_balance = 100000  # Starting with 100k
            balance = initial_balance
            units_held = 0
            trades = 0
            hold_threshold = 0.01  # Threshold to hold before selling
            
            for i in range(len(y_pred)):
                if i == 0:
                    continue
                pred_change = y_pred[i]
                actual_price = df_test['current'].iloc[i]
                
                # Sell only if predicted change is significantly negative and units held
                if pred_change < -hold_threshold and units_held > 0:
                    sell_units = min(units_held, 1)  # Sell 1 unit to reduce selling frequency
                    balance += sell_units * actual_price
                    units_held -= sell_units
                    trades += 1
                # Buy only if predicted change is significantly positive and balance allows
                elif pred_change > hold_threshold and balance > actual_price:
                    buy_units = 1  # Buy 1 unit
                    balance -= buy_units * actual_price
                    units_held += buy_units
                    trades += 1
                    
            final_value = balance + units_held * df_test['current'].iloc[-1]
            profit = final_value - initial_balance
            profit_metrics = {
                'Total_Profit': profit,
                'Profit_Percentage': (profit / initial_balance) * 100,
                'Total_Trades': trades
            }
            logger.info(f"Improved trading simulation results: {profit_metrics}")
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Directional_Accuracy': direction_correct
        }
        metrics.update(profit_metrics)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    def plot_results(self, history, y_test=None, y_pred=None):
        """
        Plot training history and predictions if available.
        
        Args:
            history: Training history object
            y_test (numpy.array): Actual test values
            y_pred (numpy.array): Predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_loss.png')
        plt.close()
        
        if y_test is not None and y_pred is not None:
            # Inverse transform for plotting
            y_test_full = np.zeros((len(y_test), len(self.feature_cols)))
            y_test_full[:, self.feature_cols.index(self.target_col)] = y_test
            y_test_inv = self.scaler.inverse_transform(y_test_full)[:, self.feature_cols.index(self.target_col)]
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_inv, label='Actual Change')
            plt.plot(y_pred, label='Predicted Change')
            plt.title('Stock Change Prediction Results')
            plt.xlabel('Time')
            plt.ylabel('Change')
            plt.legend()
            plt.savefig('prediction_results.png')
            plt.close()
            logger.info("Plots saved as training_loss.png and prediction_results.png")

def main():
    """Main function to run the stock prediction pipeline."""
    try:
        predictor = StockPredictor()
        df = predictor.load_data('synthetic_hsi_last_30_days.csv')
        
        # Preprocess and split data
        X, y = predictor.preprocess_data(df, train=True)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        df_test = df.iloc[split:]
        
        # Build and train model
        predictor.build_model()
        history = predictor.train(X_train, y_train)
        
        # Evaluate
        metrics = predictor.evaluate(X_test, y_test, df_test)
        y_pred = predictor.predict(X_test)
        
        # Plot results
        predictor.plot_results(history, y_test, y_pred)
        
        logger.info("Stock prediction pipeline completed successfully")
        print("Evaluation Metrics:", metrics)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
