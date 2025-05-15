import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_path="stock_model.joblib", sequence_length=60):
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
        self.target_col = 'change'
        
    def load_data(self, filepath):
        """
        Load and prepare stock data from CSV.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Processed dataframe
        """
        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # Convert numeric columns, replacing any non-numeric values with NaN
            numeric_cols = ['current', 'turnover', 'high', 'low', 'change', 'percent']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with null values
            df.dropna(subset=numeric_cols, inplace=True)
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df = df.sort_values('datetime')
            
            logger.info(f"Loaded data from {filepath} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_features(self, df):
        """
        Prepare features for model training/prediction.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            tuple: X (features) and y (target)
        """
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        # Use only the specified features
        X = df[self.feature_cols]
        y = df[self.target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
        
    def build_model(self, model_type='rf'):
        """
        Build the model based on specified type.
        
        Args:
            model_type (str): Type of model ('rf' for Random Forest)
        """
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        self.model = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info(f"Built {model_type.upper()} model")
        
    def train(self, X, y):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info("Model trained and saved")
        
        # Calculate validation metrics
        val_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        mse = mean_squared_error(y_val, val_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
        
    def load_trained_model(self):
        """
        Load a pre-trained model if available.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded pre-trained model from {self.model_path}")
        else:
            logger.warning("No pre-trained model found")
            self.build_model()
            
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            numpy.array: Predicted values
        """
        if self.model is None:
            self.load_trained_model()
            
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(y_pred)
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        # Create confusion matrix for directional prediction
        true_pos = np.sum((actual_direction == 1) & (pred_direction == 1))
        true_neg = np.sum((actual_direction == -1) & (pred_direction == -1))
        false_pos = np.sum((actual_direction == -1) & (pred_direction == 1))
        false_neg = np.sum((actual_direction == 1) & (pred_direction == -1))
        
        confusion_matrix = {
            'True Positive (Correctly Predicted Up)': true_pos,
            'True Negative (Correctly Predicted Down)': true_neg,
            'False Positive (Incorrectly Predicted Up)': false_pos,
            'False Negative (Incorrectly Predicted Down)': false_neg
        }
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R² Score': r2,
            'Directional Accuracy': direction_accuracy,
            'Confusion Matrix': confusion_matrix
        }
        
        logger.info(f"Test metrics: {metrics}")
        return metrics, y_pred
        
    def plot_results(self, y_test, y_pred):
        """
        Plot prediction results.
        
        Args:
            y_test: Actual test values
            y_pred: Predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual Change')
        plt.plot(y_pred, label='Predicted Change')
        plt.title('Stock Change Prediction Results')
        plt.xlabel('Time')
        plt.ylabel('Change')
        plt.legend()
        plt.savefig('prediction_results.png')
        plt.close()
        logger.info("Plot saved as prediction_results.png")

def main():
    """Main function to run the stock prediction pipeline."""
    try:
        # Initialize predictor
        predictor = StockPredictor()
        
        # Load and prepare data
        df = predictor.load_data('synthetic_hsi_last_30_days.csv')
        X, y = predictor.prepare_features(df)
        
        # Build and train model
        predictor.build_model(model_type='rf')  # Use Random Forest
        metrics = predictor.train(X, y)
        
        # Evaluate on test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_metrics, y_pred = predictor.evaluate(X_test, y_test)
        
        # Plot results
        predictor.plot_results(y_test, y_pred)
        
        logger.info("Stock prediction pipeline completed successfully")
        print("\nTraining Metrics:")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        if hasattr(predictor.model, 'oob_score_'):
            print(f"Out-of-Bag Score: {predictor.model.oob_score_:.4f}")
        
        print("\nTest Metrics:")
        print(f"MAE: {test_metrics['MAE']:.4f}")
        print(f"MSE: {test_metrics['MSE']:.4f}")
        print(f"RMSE: {test_metrics['RMSE']:.4f}")
        print(f"R² Score: {test_metrics['R² Score']:.4f}")
        print(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.2%}")
        
        print("\nConfusion Matrix for Directional Prediction:")
        conf_matrix = test_metrics['Confusion Matrix']
        print(f"True Positive (Correctly Predicted Up): {conf_matrix['True Positive (Correctly Predicted Up)']}")
        print(f"True Negative (Correctly Predicted Down): {conf_matrix['True Negative (Correctly Predicted Down)']}")
        print(f"False Positive (Incorrectly Predicted Up): {conf_matrix['False Positive (Incorrectly Predicted Up)']}")
        print(f"False Negative (Incorrectly Predicted Down): {conf_matrix['False Negative (Incorrectly Predicted Down)']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
