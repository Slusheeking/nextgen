#!/usr/bin/env python3
"""
Time Series MCP Tool

This module implements a consolidated Model Context Protocol (MCP) server for
advanced time series analysis, including technical indicators, pattern recognition,
peak/trough detection, drift detection, and forecasting using state-of-the-art models.
"""

import os
import sys
import json
import time
import importlib
from typing import Dict, List, Any, Optional

# Direct imports instead of dynamic loading
try:
    import numpy as np
except ImportError:
    np = None
    print("Warning: NumPy not installed or import failed.")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: Pandas not installed or import failed.")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

# Import libraries for technical analysis and time series modeling with proper error handling
try:
    import talib
    HAVE_TALIB = True
except ImportError:
    HAVE_TALIB = False
    talib = None
    print("Warning: TA-Lib not installed or import failed. Technical indicator features will be limited.")

# Advanced time series models
HAVE_STATSMODELS = False
sm = None
acf = None
try:
    # Need to import statsmodels directly due to its package structure
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAVE_STATSMODELS = True
except (ImportError, ModuleNotFoundError):
    print("Warning: statsmodels not installed or import failed. Advanced time series modeling will be limited.")

# Prophet forecasting
try:
    import prophet
    HAVE_PROPHET = True
except ImportError:
    HAVE_PROPHET = False
    print("Warning: Prophet not installed or import failed. Facebook Prophet forecasting will not be available.")

# TensorFlow and Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model #type: ignore
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout #type: ignore
    from tensorflow.keras.callbacks import EarlyStopping #type: ignore
    HAVE_TF = True
    print("TensorFlow successfully integrated for deep learning forecasting.")
except ImportError:
    HAVE_TF = False
    tf = None
    print("Warning: TensorFlow not installed or import failed. Deep learning forecasting will not be available.")

# GluonTS and DeepAR for advanced forecasting
try:
    # First, try to handle the numpy bool deprecation issue
    # This is a workaround for the MxNet numpy compatibility issue
    import numpy as np
    if not hasattr(np, 'bool'):
        np.bool = bool  # Set np.bool to Python's bool as a fallback
    
    # Import GluonTS components with preference for torch backend
    from gluonts.dataset.common import ListDataset
    
    # Always try to use torch backend first to avoid MxNet compatibility issues
    try:
        from gluonts.torch import DeepAREstimator
        from gluonts.torch.model.estimator import Trainer as TorchTrainer
        use_torch_trainer = True
        Trainer = TorchTrainer
        print("Using GluonTS torch trainer")
    except ImportError:
        # Fall back to MxNet trainer only if torch is not available
        try:
            # For older GluonTS versions
            from gluonts.mx.model.deepar import DeepAREstimator
            from gluonts.mx.trainer import Trainer
            use_torch_trainer = False
            print("Using GluonTS MxNet trainer")
        except ImportError:
            # Try alternative import paths for newer GluonTS versions
            try:
                from gluonts.mx.model.deepar import DeepAREstimator
                from gluonts.mx.trainer import Trainer
                use_torch_trainer = False
                print("Using GluonTS MxNet trainer (alternative import)")
            except Exception as mx_e:
                print(f"Warning: MxNet trainer unavailable: {mx_e}")
                use_torch_trainer = True
                Trainer = None
    
    from gluonts.evaluation.backtest import make_evaluation_predictions
    HAVE_GLUONTS = True
    print("GluonTS successfully integrated for advanced time series forecasting.")
except ImportError as e:
    HAVE_GLUONTS = False
    print(f"Warning: GluonTS not installed or import failed: {e}. Advanced forecasting with DeepAR will not be available.")

# Set overall advanced time series capability flag
HAVE_ADVANCED_TS = HAVE_STATSMODELS or HAVE_PROPHET or HAVE_TF or HAVE_GLUONTS

# Import libraries for peak detection and drift detection
HAVE_SCIPY = False
find_peaks = None
kendalltau = None
try:
    import scipy
    from scipy.signal import find_peaks
    from scipy.stats import kendalltau
    HAVE_SCIPY = True
except ImportError:
    print("Warning: SciPy not installed or import failed. Peak detection features will be limited.")

# Drift detection libraries
HAVE_DRIFT_DETECTION = False
try:
    # Try to import river with direct import
    import river.drift as drift
    HAVE_DRIFT_DETECTION = True
    print("River drift detection library available.")
except (ImportError, ModuleNotFoundError):
    try:
        # Alternative drift detection with alibi-detect
        from alibi_detect.cd import MMDDrift
        HAVE_DRIFT_DETECTION = True
        print("Alibi Detect drift detection library available.")
    except (ImportError, ModuleNotFoundError):
        print("Warning: Drift detection libraries not found. Using basic statistical methods.")


class TimeSeriesMCP(BaseMCPServer):
    """
    Consolidated MCP server for time series analysis.
    Handles technical indicators, pattern recognition, peak/drift detection, and forecasting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Time Series MCP server.

        Args:
            config: Optional configuration dictionary. If None, loads from
                  config/time_series_mcp/time_series_mcp_config.json
        """
        if config is None:
            config_path = os.path.join("config", "time_series_mcp", "time_series_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                print(f"Warning: Config file not found at {config_path}. Using default settings.")
                config = {}

        super().__init__(name="time_series_mcp", config=config)

        self._configure_from_config()
        self._initialize_models() # Placeholder for loading time series models
        self._register_tools()

        self.logger.info("Time Series MCP initialized successfully")

    def _configure_from_config(self):
        """Extract configuration values."""
        # Technical Indicator settings
        indicator_config = self.config.get("indicators", {})
        self.default_indicators = indicator_config.get("defaults", ["SMA", "EMA", "RSI", "MACD"])
        self.indicator_params = indicator_config.get("params", {
            "SMA": {"timeperiod": 20},
            "EMA": {"timeperiod": 20},
            "RSI": {"timeperiod": 14},
            "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
        })

        # Peak Detection settings
        peak_config = self.config.get("peak_detection", {})
        self.peak_prominence = peak_config.get("prominence", 0.01) # Example: 1% prominence
        self.peak_distance = peak_config.get("distance", 5) # Minimum 5 periods apart

        # Drift Detection settings
        drift_config = self.config.get("drift_detection", {})
        self.drift_window_size = drift_config.get("window_size", 30)
        self.drift_threshold = drift_config.get("threshold", 0.05) # Example threshold

        # Forecasting settings
        forecast_config = self.config.get("forecasting", {})
        self.forecast_horizon = forecast_config.get("horizon", 5) # Default 5 periods ahead
        self.forecast_model_path = forecast_config.get("model_path", None) # Path for advanced models

        self.logger.info("Time Series MCP configuration loaded",
                       default_indicators=self.default_indicators,
                       forecast_horizon=self.forecast_horizon)

    def _initialize_models(self):
        """
        Initialize advanced time series models and detectors if configured.
        This includes loading pre-trained forecasting models, initializing
        drift detectors, and setting up any other required statistical models.
        """
        # Initialize forecasting models
        self.forecast_model = None
        self.forecast_models = {}  # Dictionary to store multiple models
        
        # Check if advanced time series libraries are available
        if HAVE_ADVANCED_TS and self.forecast_model_path:
            try:
                self.logger.info(f"Loading advanced forecast model from {self.forecast_model_path}")
                
                # Load appropriate model based on file extension and available libraries
                if os.path.isdir(self.forecast_model_path):
                    # Directory of models - handle multiple model files
                    model_files = [f for f in os.listdir(self.forecast_model_path) 
                                  if f.endswith(('.model', '.pkl', '.h5', '.keras', '.pb'))]
                    
                    for model_file in model_files:
                        model_name = os.path.splitext(model_file)[0]
                        model_path = os.path.join(self.forecast_model_path, model_file)
                        
                        try:
                            # Handle different model types based on extension
                            if model_file.endswith(('.h5', '.keras')) and HAVE_TF:
                                # Load TensorFlow/Keras model
                                self.forecast_models[model_name] = load_model(model_path)
                                self.logger.info(f"Loaded TensorFlow/Keras model '{model_name}' from {model_file}")
                            
                            elif model_file.endswith('.pb') and HAVE_TF:
                                # Load TensorFlow SavedModel format
                                self.forecast_models[model_name] = tf.saved_model.load(model_path)
                                self.logger.info(f"Loaded TensorFlow SavedModel '{model_name}' from {model_file}")
                            
                            elif model_file.endswith('.pkl') and HAVE_STATSMODELS:
                                # Load pickle format (likely statsmodels or sklearn)
                                import pickle
                                with open(model_path, 'rb') as f:
                                    self.forecast_models[model_name] = pickle.load(f)
                                self.logger.info(f"Loaded pickled model '{model_name}' from {model_file}")
                                
                            elif HAVE_GLUONTS and model_file.endswith(('.json', '.params')):
                                # Handle GluonTS models if path contains model info
                                if model_file.endswith('.json') and os.path.exists(model_path.replace('.json', '.params')):
                                    try:
                                        # Updated import path for Predictor
                                        try:
                                            from gluonts.torch.model.predictor import Predictor
                                        except ImportError:
                                            from gluonts.model.predictor import Predictor
                                            
                                        
                                        self.forecast_models[model_name] = Predictor.deserialize(
                                            os.path.dirname(model_path)
                                        )
                                        self.logger.info(f"Loaded GluonTS model '{model_name}' from {model_file}")
                                    except Exception as gluon_e:
                                        self.logger.error(f"Failed to load GluonTS model '{model_name}': {gluon_e}")
                                        
                            else:
                                self.logger.warning(f"Unknown model type for '{model_file}', couldn't load automatically")
                                
                        except Exception as model_e:
                            self.logger.error(f"Failed to load model '{model_name}': {model_e}")
                
                else:
                    # Single model file
                    file_ext = os.path.splitext(self.forecast_model_path)[1].lower()
                    model_name = os.path.basename(os.path.splitext(self.forecast_model_path)[0])
                    
                    try:
                        if file_ext in ('.h5', '.keras') and HAVE_TF:
                            self.forecast_model = load_model(self.forecast_model_path)
                            self.logger.info(f"Loaded TensorFlow model from {self.forecast_model_path}")
                            
                        elif file_ext == '.pb' and HAVE_TF:
                            self.forecast_model = tf.saved_model.load(self.forecast_model_path)
                            self.logger.info(f"Loaded TensorFlow SavedModel from {self.forecast_model_path}")
                            
                        elif file_ext == '.pkl' and HAVE_STATSMODELS:
                            import pickle
                            with open(self.forecast_model_path, 'rb') as f:
                                self.forecast_model = pickle.load(f)
                            self.logger.info(f"Loaded pickled model from {self.forecast_model_path}")
                            
                        elif HAVE_GLUONTS and file_ext in ('.json', '.params'):
                            # GluonTS model directory
                            model_dir = os.path.dirname(self.forecast_model_path)
                            # Use updated import path for Predictor
                            try:
                                from gluonts.torch.model.predictor import Predictor
                            except ImportError:
                                from gluonts.model.predictor import Predictor
                            self.forecast_model = Predictor.deserialize(model_dir)
                            self.logger.info(f"Loaded GluonTS model from {model_dir}")
                            
                        else:
                            self.logger.warning(f"Unknown model type '{file_ext}', couldn't load automatically")
                    except Exception as e:
                        self.logger.error(f"Failed to load model from {self.forecast_model_path}: {e}")
                
                # Count total models loaded
                total_models = len(self.forecast_models) + (1 if self.forecast_model else 0)
                self.logger.info(f"Advanced forecast models initialized: {total_models} models loaded")
                
            except Exception as e:
                self.logger.error(f"Failed to load forecast models: {e}", exc_info=True)
        else:
            self.logger.info("Using basic forecasting methods (no advanced models loaded).")
            
            # Initialize statistical models that don't require external libraries
            try:
                # Initialize simple exponential smoothing models with different alpha values
                self.forecast_models['exp_smooth'] = {
                    'alpha_low': 0.1,    # Low smoothing (more weight to history)
                    'alpha_med': 0.3,    # Medium smoothing
                    'alpha_high': 0.7    # High smoothing (more weight to recent values)
                }
                
                # Initialize simple moving average models with different window sizes
                self.forecast_models['sma'] = {
                    'window_sizes': [5, 10, 20, 50]
                }
                
                self.logger.info("Initialized basic statistical forecasting models")
            except Exception as e:
                self.logger.error(f"Failed to initialize basic statistical models: {e}")

        # Initialize drift detectors
        self.drift_detectors = {}  # Store detectors per series/metric
        
        if HAVE_DRIFT_DETECTION:
            self.logger.info("Drift detection library available, initializing detectors")
            try:
                # Initialize drift detectors with different sensitivity levels
                if 'river' in sys.modules:
                    # River's ADWIN (Adaptive Windowing) detector
                    self.drift_detector_class = drift.ADWIN
                    self.drift_detectors['high_sensitivity'] = self.drift_detector_class(delta=0.002)
                    self.drift_detectors['medium_sensitivity'] = self.drift_detector_class(delta=0.005)
                    self.drift_detectors['low_sensitivity'] = self.drift_detector_class(delta=0.01)
                    self.logger.info("Initialized River ADWIN drift detectors with multiple sensitivity levels")
                elif 'alibi_detect' in sys.modules:
                    # Alibi Detect's MMD (Maximum Mean Discrepancy) drift detector
                    from alibi_detect.cd import MMDDrift
                    # Create reference data from normal distribution for initial state
                    reference_data = np.random.normal(0, 1, (100, 1))
                    self.drift_detectors['high_sensitivity'] = MMDDrift(
                        reference_data, p_val=0.01, kernel='rbf'
                    )
                    self.drift_detectors['medium_sensitivity'] = MMDDrift(
                        reference_data, p_val=0.05, kernel='rbf'
                    )
                    self.drift_detectors['low_sensitivity'] = MMDDrift(
                        reference_data, p_val=0.1, kernel='rbf'
                    )
                    self.logger.info("Initialized Alibi Detect MMD drift detectors with multiple sensitivity levels")
            except Exception as e:
                self.logger.error(f"Failed to initialize drift detectors: {e}")
        else:
            self.logger.warning("Drift detection library not available, using basic methods")
            # Initialize basic drift detection parameters for different sensitivity levels
            self.drift_detectors['high_sensitivity'] = {
                'window_size': 20,
                'z_threshold': 1.5
            }
            self.drift_detectors['medium_sensitivity'] = {
                'window_size': 20,
                'z_threshold': 2.0
            }
            self.drift_detectors['low_sensitivity'] = {
                'window_size': 20,
                'z_threshold': 3.0
            }
            self.logger.info("Initialized basic drift detection parameters")
        
        # Initialize pattern recognition models if needed
        self.pattern_models = {}
        if HAVE_TALIB:
            self.logger.info("TA-Lib available for pattern recognition")
            # TA-Lib patterns are already available through the library
            # No additional initialization needed
        else:
            self.logger.warning("TA-Lib not available, pattern recognition will be limited")
            # Could initialize basic pattern templates here


    def _register_tools(self):
        """Register unified time series analysis tools."""
        self.register_tool(
            self.calculate_indicators,
            "calculate_indicators",
            "Calculate technical indicators for time series data."
        )
        self.register_tool(
            self.detect_patterns,
            "detect_patterns",
            "Detect common chart patterns (e.g., head and shoulders, flags)."
        )
        self.register_tool(
            self.detect_peaks_troughs,
            "detect_peaks_troughs",
            "Detect significant peaks and troughs in time series data."
        )
        self.register_tool(
            self.detect_support_resistance,
            "detect_support_resistance",
            "Detect support and resistance levels in price data."
        )
        self.register_tool(
            self.detect_drift,
            "detect_drift",
            "Detect statistical drift or regime changes in time series data."
        )
        self.register_tool(
            self.analyze_correlation,
            "analyze_correlation",
            "Analyze correlation between multiple time series."
        )
        self.register_tool(
            self.forecast_series,
            "forecast_series",
            "Generate forecasts for time series data."
        )

    def _validate_dataframe(self, data: Any) -> Optional["pd.DataFrame"]:
        """Validate and convert input data to a Pandas DataFrame with expected columns."""
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            try:
                df = pd.DataFrame(data)
                # Ensure standard OHLCV columns exist (case-insensitive check)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                df.columns = df.columns.str.lower() # Standardize column names
                if not all(col in df.columns for col in required_cols):
                     self.logger.error(f"Input data missing required OHLCV columns. Found: {list(df.columns)}")
                     return None
                # Convert relevant columns to numeric, coercing errors
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=required_cols, inplace=True) # Drop rows with NaN in essential columns
                # Ensure time index if available
                if 't' in df.columns: # Polygon style timestamp
                     df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                     df = df.set_index('timestamp')
                elif 'date' in df.columns:
                     df['timestamp'] = pd.to_datetime(df['date'])
                     df = df.set_index('timestamp')
                elif df.index.dtype == 'int64': # If index is timestamp int
                     df.index = pd.to_datetime(df.index, unit='ms') # Assume ms

                if not isinstance(df.index, pd.DatetimeIndex):
                     self.logger.warning("DataFrame index is not DatetimeIndex. Some functions might fail.")

                return df
            except Exception as e:
                self.logger.error(f"Failed to convert input data to DataFrame: {e}", exc_info=True)
                return None
        elif isinstance(data, pd.DataFrame):
             # Basic validation for existing DataFrame
             df = data.copy()
             df.columns = df.columns.str.lower()
             required_cols = ['open', 'high', 'low', 'close', 'volume']
             if not all(col in df.columns for col in required_cols):
                  self.logger.error(f"Input DataFrame missing required OHLCV columns. Found: {list(df.columns)}")
                  return None
             return df
        else:
            self.logger.error(f"Invalid input data type: {type(data)}. Expected list of dicts or DataFrame.")
            return None

    # --- Tool Implementations ---

    def calculate_indicators(self, data: Any, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate technical indicators."""
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None:
            return {"error": "Invalid input data format for indicators."}

        indicators_to_calc = indicators or self.default_indicators
        self.logger.info("Calculating indicators", indicators=indicators_to_calc, data_points=len(df))
        results = {}

        if not HAVE_TALIB:
            return {"error": "TA-Lib not installed. Cannot calculate indicators."}

        # Ensure required columns are present and numeric
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)

        for indicator in indicators_to_calc:
            try:
                params = self.indicator_params.get(indicator, {})
                func_name = indicator.upper() # TA-Lib functions are uppercase

                if hasattr(talib, func_name):
                    func = getattr(talib, func_name)
                    # Pass required data based on indicator type
                    # This needs a mapping or more robust logic
                    if indicator in ["SMA", "EMA", "RSI"]:
                        results[indicator] = func(close, **params).tolist()
                    elif indicator == "MACD":
                        # MACD returns macd, macdsignal, macdhist
                        macd, signal, hist = func(close, **params)
                        results[indicator] = {
                            "macd": macd.tolist(),
                            "signal": signal.tolist(),
                            "hist": hist.tolist()
                        }
                    elif indicator == "BBANDS":
                         upper, middle, lower = func(close, **params)
                         results[indicator] = {"upper": upper.tolist(), "middle": middle.tolist(), "lower": lower.tolist()}
                    # Add more indicators here...
                    else:
                         self.logger.warning(f"Indicator '{indicator}' calculation logic not fully implemented.")
                         # Try a generic call (might fail if inputs are wrong)
                         try:
                              results[indicator] = func(close, **params).tolist()
                         except Exception as func_err:
                              self.logger.error(f"Failed to call TA-Lib function {func_name}: {func_err}")
                              results[indicator] = {"error": f"Calculation failed for {indicator}"}

                else:
                    self.logger.warning(f"Indicator '{indicator}' not found in TA-Lib.")
                    results[indicator] = {"error": "Indicator not supported"}
            except Exception as e:
                self.logger.error(f"Error calculating indicator {indicator}: {e}", exc_info=True)
                results[indicator] = {"error": str(e)}

        processing_time = time.time() - start_time
        self.logger.timing("calculate_indicators_time_ms", processing_time * 1000, indicator_count=len(indicators_to_calc))
        return {"indicators": results, "processing_time": processing_time}


    def detect_patterns(self, data: Any) -> Dict[str, Any]:
        """Detect common chart patterns."""
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None:
            return {"error": "Invalid input data format for pattern detection."}

        self.logger.info("Detecting chart patterns", data_points=len(df))
        patterns = {}

        if not HAVE_TALIB:
            return {"error": "TA-Lib not installed. Cannot detect patterns."}

        # TA-Lib pattern recognition functions (CDL*) operate on OHLC data
        op = df['open'].values.astype(float)
        hi = df['high'].values.astype(float)
        lo = df['low'].values.astype(float)
        cl = df['close'].values.astype(float)

        # Iterate through all TA-Lib pattern functions (prefixed with CDL)
        for func_name in dir(talib):
            if func_name.startswith('CDL'):
                try:
                    func = getattr(talib, func_name)
                    # The result is an array where non-zero values indicate pattern occurrence
                    # 100 = bullish pattern, -100 = bearish pattern at that index
                    result = func(op, hi, lo, cl)
                    occurrences = np.where(result != 0)[0] # Get indices where pattern occurs
                    if len(occurrences) > 0:
                        # Store indices and pattern type (bullish/bearish)
                        patterns[func_name] = [{"index": int(idx), "type": 100 if result[idx] > 0 else -100}
                                               for idx in occurrences]
                except Exception as e:
                    self.logger.warning(f"Error executing TA-Lib pattern function {func_name}: {e}")

        processing_time = time.time() - start_time
        self.logger.timing("detect_patterns_time_ms", processing_time * 1000, pattern_count=len(patterns))
        return {"patterns": patterns, "processing_time": processing_time}


    def detect_peaks_troughs(self, data: Any, column: str = 'close') -> Dict[str, Any]:
        """Detect significant peaks and troughs."""
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None or column not in df.columns:
            return {"error": f"Invalid input data or column '{column}' not found."}

        self.logger.info(f"Detecting peaks and troughs in column '{column}'", data_points=len(df))

        if not HAVE_SCIPY:
            return {"error": "SciPy not installed. Cannot detect peaks/troughs."}

        series = df[column].values.astype(float)
        try:
            # Calculate prominence relative to the series range
            series_range = np.ptp(series) # Peak-to-peak range
            if series_range == 0: # Avoid division by zero for flat series
                 prominence_value = None
            else:
                 prominence_value = self.peak_prominence * series_range

            # Find peaks
            peaks, peak_props = find_peaks(series, prominence=prominence_value, distance=self.peak_distance)  # type: ignore
            # Find troughs (by inverting the series)
            troughs, trough_props = find_peaks(-series, prominence=prominence_value, distance=self.peak_distance)  # type: ignore

            results = {
                "peaks": [{"index": int(p), "value": series[p], "prominence": peak_props['prominences'][i]}
                          for i, p in enumerate(peaks)],
                "troughs": [{"index": int(t), "value": series[t], "prominence": trough_props['prominences'][i]}
                            for i, t in enumerate(troughs)]
            }
            processing_time = time.time() - start_time
            self.logger.timing("detect_peaks_troughs_time_ms", processing_time * 1000, peak_count=len(peaks), trough_count=len(troughs))
            return {"extrema": results, "processing_time": processing_time}

        except Exception as e:
            self.logger.error(f"Error detecting peaks/troughs: {e}", exc_info=True)
            return {"error": str(e)}
            
    def detect_support_resistance(self, data: Any, method: str = 'peaks', 
                                 column: str = 'close', lookback: int = 100, 
                                 cluster_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Detect support and resistance levels in price data using various methods.
        
        Args:
            data: Time series data (list of dicts or DataFrame)
            method: Method to use ('peaks', 'histogram', 'fibonacci', 'pivot')
            column: Column to analyze for support/resistance
            lookback: Number of periods to look back
            cluster_threshold: Threshold for clustering price levels (as % of price range)
            
        Returns:
            Dictionary with support and resistance levels
        """
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None or column not in df.columns:
            return {"error": f"Invalid input data or column '{column}' not found."}

        self.logger.info(f"Detecting support/resistance in column '{column}' using method '{method}'", 
                        data_points=len(df), lookback=lookback)

        # Use only the lookback period if specified
        if lookback and lookback < len(df):
            df = df.iloc[-lookback:]
            
        series = df[column].values.astype(float)
        
        try:
            # Different methods for detecting support/resistance
            if method == 'peaks':
                levels = self._detect_sr_using_peaks(df, column, cluster_threshold)
            elif method == 'histogram':
                levels = self._detect_sr_using_histogram(df, column)
            elif method == 'fibonacci':
                levels = self._detect_sr_using_fibonacci(df, column)
            elif method == 'pivot':
                levels = self._detect_sr_using_pivot_points(df)
            else:
                return {"error": f"Unsupported method: {method}. Use 'peaks', 'histogram', 'fibonacci', or 'pivot'."}
                
            # Add current price context
            current_price = float(series[-1]) if len(series) > 0 else None
            
            # Identify nearest support and resistance levels to current price
            if current_price and levels["support"] and levels["resistance"]:
                # Find nearest support (below current price)
                supports_below = [s for s in levels["support"] if s["value"] < current_price]
                if supports_below:
                    nearest_support = max(supports_below, key=lambda x: x["value"])
                    levels["nearest_support"] = nearest_support
                
                # Find nearest resistance (above current price)
                resistances_above = [r for r in levels["resistance"] if r["value"] > current_price]
                if resistances_above:
                    nearest_resistance = min(resistances_above, key=lambda x: x["value"])
                    levels["nearest_resistance"] = nearest_resistance
            
            # Add current price context
            levels["current_price"] = current_price
            
            # Calculate distance to nearest levels as percentage
            if current_price:
                if "nearest_support" in levels:
                    support_value = levels["nearest_support"]["value"]
                    levels["nearest_support"]["distance_pct"] = (current_price - support_value) / current_price * 100
                    
                if "nearest_resistance" in levels:
                    resistance_value = levels["nearest_resistance"]["value"]
                    levels["nearest_resistance"]["distance_pct"] = (resistance_value - current_price) / current_price * 100
            
            processing_time = time.time() - start_time
            self.logger.timing("detect_support_resistance_time_ms", processing_time * 1000, 
                              method=method, support_count=len(levels["support"]), 
                              resistance_count=len(levels["resistance"]))
            
            return {
                "levels": levels,
                "method": method,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _detect_sr_using_peaks(self, df: "pd.DataFrame", column: str = 'close',
                               cluster_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Detect support and resistance levels using peak/trough analysis.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            cluster_threshold: Threshold for clustering price levels (as % of price range)
            
        Returns:
            Dictionary with support and resistance levels
        """
        series = df[column].values.astype(float)
        
        if not HAVE_SCIPY:
            return {"error": "SciPy not installed. Cannot detect peaks/troughs."}
            
        # Calculate prominence based on price range
        series_range = np.ptp(series)
        if series_range == 0:  # Avoid division by zero
            prominence_value = None
        else:
            # Use a more aggressive prominence for S/R detection
            prominence_value = self.peak_prominence * 1.5 * series_range
            
        # Find peaks (potential resistance)
        peaks, peak_props = find_peaks(series, prominence=prominence_value, distance=self.peak_distance)
        
        # Find troughs (potential support)
        troughs, trough_props = find_peaks(-series, prominence=prominence_value, distance=self.peak_distance)
        
        # Convert to price levels
        resistance_levels = [{"index": int(p), "value": float(series[p]), 
                             "strength": float(peak_props['prominences'][i])}
                            for i, p in enumerate(peaks)]
        
        support_levels = [{"index": int(t), "value": float(series[t]), 
                          "strength": float(trough_props['prominences'][i])}
                         for i, t in enumerate(troughs)]
        
        # Cluster similar levels
        resistance_levels = self._cluster_price_levels(resistance_levels, cluster_threshold * series_range)
        support_levels = self._cluster_price_levels(support_levels, cluster_threshold * series_range)
        
        # Sort by price
        resistance_levels.sort(key=lambda x: x["value"])
        support_levels.sort(key=lambda x: x["value"])
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def _detect_sr_using_histogram(self, df: "pd.DataFrame", column: str = 'close') -> Dict[str, Any]:
        """
        Detect support and resistance levels using price histogram analysis.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            
        Returns:
            Dictionary with support and resistance levels
        """
        series = df[column].values.astype(float)
        
        # Create histogram
        hist_bins = min(100, len(series) // 5)  # Reasonable number of bins
        if hist_bins < 10:
            hist_bins = 10  # Minimum bins
            
        hist, bin_edges = np.histogram(series, bins=hist_bins)
        
        # Find local maxima in histogram (price levels with high frequency)
        if HAVE_SCIPY:
            # Use scipy's find_peaks for better detection
            peaks, _ = find_peaks(hist, height=np.mean(hist), distance=2)
            
            # Convert peak indices to price levels
            levels = []
            for peak_idx in peaks:
                price_level = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                strength = hist[peak_idx] / np.max(hist)  # Normalize strength
                levels.append({
                    "value": float(price_level),
                    "strength": float(strength),
                    "count": int(hist[peak_idx])
                })
        else:
            # Simple local maxima detection
            levels = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                    price_level = (bin_edges[i] + bin_edges[i + 1]) / 2
                    strength = hist[i] / np.max(hist)  # Normalize strength
                    levels.append({
                        "value": float(price_level),
                        "strength": float(strength),
                        "count": int(hist[i])
                    })
        
        # Sort levels by price
        levels.sort(key=lambda x: x["value"])
        
        # Separate into support and resistance based on current price
        current_price = series[-1]
        support_levels = [level for level in levels if level["value"] < current_price]
        resistance_levels = [level for level in levels if level["value"] > current_price]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def _detect_sr_using_fibonacci(self, df: "pd.DataFrame", column: str = 'close') -> Dict[str, Any]:
        """
        Detect support and resistance levels using Fibonacci retracement levels.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            
        Returns:
            Dictionary with support and resistance levels
        """
        series = df[column].values.astype(float)
        
        # Need at least some data points
        if len(series) < 10:
            return {"support": [], "resistance": []}
        
        # Find highest high and lowest low in the period
        highest = np.max(series)
        lowest = np.min(series)
        price_range = highest - lowest
        
        # Standard Fibonacci retracement levels
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        # Calculate price at each level
        levels = []
        
        # Determine if we're in an uptrend or downtrend
        # Simple method: compare first half average to second half average
        mid_point = len(series) // 2
        first_half_avg = np.mean(series[:mid_point])
        second_half_avg = np.mean(series[mid_point:])
        
        if second_half_avg > first_half_avg:
            # Uptrend: calculate retracements from bottom to top
            for fib in fib_levels:
                price = lowest + (price_range * fib)
                levels.append({
                    "value": float(price),
                    "level": float(fib),
                    "type": "retracement"
                })
                
            # Add extension levels for uptrend
            extensions = [1.272, 1.618, 2.0, 2.618]
            for ext in extensions:
                price = lowest + (price_range * ext)
                levels.append({
                    "value": float(price),
                    "level": float(ext),
                    "type": "extension"
                })
        else:
            # Downtrend: calculate retracements from top to bottom
            for fib in fib_levels:
                price = highest - (price_range * fib)
                levels.append({
                    "value": float(price),
                    "level": float(fib),
                    "type": "retracement"
                })
                
            # Add extension levels for downtrend
            extensions = [1.272, 1.618, 2.0, 2.618]
            for ext in extensions:
                price = highest - (price_range * ext)
                levels.append({
                    "value": float(price),
                    "level": float(ext),
                    "type": "extension"
                })
        
        # Sort levels by price
        levels.sort(key=lambda x: x["value"])
        
        # Separate into support and resistance based on current price
        current_price = series[-1]
        support_levels = [level for level in levels if level["value"] < current_price]
        resistance_levels = [level for level in levels if level["value"] > current_price]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels,
            "trend": "uptrend" if second_half_avg > first_half_avg else "downtrend",
            "highest": float(highest),
            "lowest": float(lowest)
        }
    
    def _detect_sr_using_pivot_points(self, df: "pd.DataFrame") -> Dict[str, Any]:
        """
        Detect support and resistance levels using pivot point analysis.
        
        Args:
            df: DataFrame with price data (must have OHLC columns)
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Ensure we have required columns
        required = ['high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return {"error": "Missing required columns for pivot point calculation"}
        
        # Get the most recent complete period (usually a day)
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Calculate pivot point (PP)
        pp = (high + low + close) / 3
        
        # Calculate support levels
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        
        # Calculate resistance levels
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        
        # Create level objects
        support_levels = [
            {"value": float(s1), "level": "S1", "type": "pivot"},
            {"value": float(s2), "level": "S2", "type": "pivot"},
            {"value": float(s3), "level": "S3", "type": "pivot"}
        ]
        
        resistance_levels = [
            {"value": float(r1), "level": "R1", "type": "pivot"},
            {"value": float(r2), "level": "R2", "type": "pivot"},
            {"value": float(r3), "level": "R3", "type": "pivot"}
        ]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels,
            "pivot": float(pp),
            "high": float(high),
            "low": float(low),
            "close": float(close)
        }
    
    def _cluster_price_levels(self, levels: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Cluster similar price levels to avoid redundancy.
        
        Args:
            levels: List of price level dictionaries
            threshold: Maximum difference to consider levels as the same cluster
            
        Returns:
            List of clustered price levels
        """
        if not levels:
            return []
            
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x["value"])
        
        # Cluster similar levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_level = sorted_levels[i]
            prev_level = sorted_levels[i-1]
            
            # If this level is close to the previous one, add to current cluster
            if abs(current_level["value"] - prev_level["value"]) <= threshold:
                current_cluster.append(current_level)
            else:
                # Process the completed cluster
                if current_cluster:
                    # Average the values in the cluster
                    avg_value = sum(level["value"] for level in current_cluster) / len(current_cluster)
                    # Sum the strengths
                    total_strength = sum(level.get("strength", 1.0) for level in current_cluster)
                    # Get the strongest index
                    if "index" in current_cluster[0]:
                        indices = [level["index"] for level in current_cluster]
                        # Use the index of the strongest level
                        strongest_idx = max(range(len(current_cluster)), 
                                          key=lambda i: current_cluster[i].get("strength", 1.0))
                        index = current_cluster[strongest_idx]["index"]
                    else:
                        index = None
                    
                    # Create a new level representing the cluster
                    cluster_level = {
                        "value": float(avg_value),
                        "strength": float(total_strength),
                        "count": len(current_cluster)
                    }
                    
                    if index is not None:
                        cluster_level["index"] = index
                        
                    clusters.append(cluster_level)
                
                # Start a new cluster with the current level
                current_cluster = [current_level]
        
        # Don't forget the last cluster
        if current_cluster:
            avg_value = sum(level["value"] for level in current_cluster) / len(current_cluster)
            total_strength = sum(level.get("strength", 1.0) for level in current_cluster)
            
            if "index" in current_cluster[0]:
                indices = [level["index"] for level in current_cluster]
                strongest_idx = max(range(len(current_cluster)), 
                                  key=lambda i: current_cluster[i].get("strength", 1.0))
                index = current_cluster[strongest_idx]["index"]
            else:
                index = None
                
            cluster_level = {
                "value": float(avg_value),
                "strength": float(total_strength),
                "count": len(current_cluster)
            }
            
            if index is not None:
                cluster_level["index"] = index
                
            clusters.append(cluster_level)
        
        return clusters


    def detect_drift(self, data: Any, column: str = 'close', lookback_days: int = 30, 
                    window_size: int = 20, z_threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect statistical drift or regime changes in time series data.
        
        Args:
            data: Time series data (list of dicts or DataFrame)
            column: Column to analyze for drift
            lookback_days: Number of days to look back for analysis
            window_size: Size of the moving window for statistics
            z_threshold: Z-score threshold for drift detection
            
        Returns:
            Dictionary with drift analysis results
        """
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None or column not in df.columns:
            return {"error": f"Invalid input data or column '{column}' not found."}

        self.logger.info(f"Detecting drift in column '{column}'", 
                        data_points=len(df), 
                        window_size=window_size, 
                        z_threshold=z_threshold)

        series = df[column].values.astype(float)
        drift_points = []  # type: ignore

        try:
            # Use the provided window size or default from config
            window_size = window_size or self.drift_window_size
            if len(series) < window_size * 2:
                return {"error": f"Not enough data points for drift detection. Need at least {window_size * 2} points."}
            
            # Calculate moving mean and standard deviation
            rolling_mean = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
            
            # Calculate z-scores for each point compared to previous window
            z_scores = []
            for i in range(window_size, len(series)):
                window = series[i-window_size:i]
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:  # Avoid division by zero
                    z_score = (series[i] - mean) / std
                    z_scores.append((i, z_score))
                else:
                    z_scores.append((i, 0))
            
            # Detect points where z-score exceeds threshold
            for idx, z_score in z_scores:
                if abs(z_score) > z_threshold:
                    self.logger.warning(f"Drift detected in '{column}' at index {idx} (z-score: {z_score:.2f})")
                    drift_points.append({
                        "index": int(idx), 
                        "value": float(series[idx]), 
                        "z_score": float(z_score),
                        "timestamp": df.index[idx].isoformat() if isinstance(df.index, pd.DatetimeIndex) else None
                    })

            # Enhanced analysis: Determine overall trend and regime
            trend = "neutral"
            if len(series) >= 2:
                # Simple trend detection based on start vs end
                start_avg = np.mean(series[:min(window_size, len(series)//3)])
                end_avg = np.mean(series[-min(window_size, len(series)//3):])
                
                if end_avg > start_avg * 1.05:  # 5% increase
                    trend = "bullish"
                elif end_avg < start_avg * 0.95:  # 5% decrease
                    trend = "bearish"
                
                # Check for volatility regime
                start_vol = np.std(series[:min(window_size, len(series)//3)])
                end_vol = np.std(series[-min(window_size, len(series)//3):])
                
                if end_vol > start_vol * 1.5:  # 50% increase in volatility
                    trend += "_volatile"
                elif end_vol < start_vol * 0.75:  # 25% decrease in volatility
                    trend += "_stable"
            
            # Detect change points using CUSUM method
            change_points = []
            if len(series) > window_size:
                # Cumulative sum method for change point detection
                s_pos = np.zeros(len(series))
                s_neg = np.zeros(len(series))
                
                # Calculate mean of first window as reference
                mean_0 = np.mean(series[:window_size])
                std_0 = np.std(series[:window_size])
                
                if std_0 > 0:  # Avoid division by zero
                    # Standardize series
                    std_series = (series - mean_0) / std_0
                    
                    # CUSUM calculation
                    for i in range(1, len(std_series)):
                        s_pos[i] = max(0, s_pos[i-1] + std_series[i] - 0.5)  # Detect upward shifts
                        s_neg[i] = max(0, s_neg[i-1] - std_series[i] - 0.5)  # Detect downward shifts
                    
                    # Find change points where CUSUM exceeds threshold
                    threshold = 5.0  # Standard threshold for CUSUM
                    cp_pos = np.where(s_pos > threshold)[0]
                    cp_neg = np.where(s_neg > threshold)[0]
                    
                    # Combine and sort change points
                    all_cp = np.unique(np.concatenate((cp_pos, cp_neg)))
                    
                    for cp in all_cp:
                        if cp >= window_size:  # Skip points in initial window
                            direction = "up" if cp in cp_pos else "down"
                            change_points.append({
                                "index": int(cp),
                                "value": float(series[cp]),
                                "direction": direction,
                                "timestamp": df.index[cp].isoformat() if isinstance(df.index, pd.DatetimeIndex) else None
                            })

            if not drift_points and not change_points:
                self.logger.info(f"No significant drift detected in '{column}'.")

            # Determine if we're in a new regime
            regime_change = False
            if len(drift_points) > 0 or len(change_points) > 0:
                # Check if any drift points are in the last quarter of the data
                recent_threshold = max(len(series) - window_size, len(series) * 3 // 4)
                recent_drifts = [p for p in drift_points if p["index"] >= recent_threshold]
                recent_changes = [p for p in change_points if p["index"] >= recent_threshold]
                
                regime_change = len(recent_drifts) > 0 or len(recent_changes) > 0

            processing_time = time.time() - start_time
            self.logger.timing("detect_drift_time_ms", processing_time * 1000, 
                              drift_count=len(drift_points),
                              change_point_count=len(change_points))
            
            return {
                "drift_points": drift_points,
                "change_points": change_points,
                "trend": trend,
                "regime_change": regime_change,
                "processing_time": processing_time
            }

        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}", exc_info=True)
            return {"error": str(e)}


    def analyze_correlation(self, data: Dict[str, List[float]], window: int = 20) -> Dict[str, Any]:
        """Analyze correlation between multiple time series."""
        start_time = time.time()
        if not isinstance(data, dict) or not data:
            return {"error": "Invalid input data format. Expected dict of lists."}

        self.logger.info(f"Analyzing correlation for {len(data)} series", window=window)

        try:
            df = pd.DataFrame(data)
            # Calculate rolling correlation (example: pairwise correlation matrix over a window)
            rolling_corr = df.rolling(window=window).corr()  # type: ignore
            # Get the latest correlation matrix
            latest_corr_matrix = rolling_corr.iloc[-len(df.columns):] # Get the last N rows

            processing_time = time.time() - start_time
            self.logger.timing("analyze_correlation_time_ms", processing_time * 1000, series_count=len(data))
            # Convert DataFrame to dict for JSON serialization
            return {"correlation_matrix": latest_corr_matrix.to_dict(), "processing_time": processing_time}

        except Exception as e:
            self.logger.error(f"Error analyzing correlation: {e}", exc_info=True)
            return {"error": str(e)}


    def forecast_series(self, data: Any, column: str = 'close', horizon: Optional[int] = None, 
                       method: str = 'auto', seasonality: int = 0) -> Dict[str, Any]:
        """
        Generate forecasts for time series data using multiple methods.
        
        Args:
            data: Time series data (list of dicts or DataFrame)
            column: Column to forecast
            horizon: Number of periods to forecast
            method: Forecasting method ('auto', 'arima', 'ets', 'prophet', etc.)
            seasonality: Seasonality period (0 for auto-detection)
            
        Returns:
            Dictionary with forecast results from multiple methods
        """
        start_time = time.time()
        df = self._validate_dataframe(data)
        if df is None or column not in df.columns:
            return {"error": f"Invalid input data or column '{column}' not found."}

        forecast_horizon = horizon or self.forecast_horizon
        self.logger.info(f"Forecasting column '{column}' for {forecast_horizon} steps", 
                        data_points=len(df), method=method)

        try:
            # Extract the series to forecast
            series = df[column]
            if len(series) < 5:  # Need some data for basic forecast
                return {"error": "Not enough data points for forecast (minimum 5 required)."}
            
            # Auto-detect seasonality if not specified
            detected_seasonality = seasonality
            if seasonality == 0 and len(series) >= 20:
                # Simple seasonality detection using autocorrelation
                try:
                    from statsmodels.tsa.stattools import acf
                    acf_values = acf(series, nlags=min(len(series) // 2, 50))
                    # Find peaks in autocorrelation
                    peaks = []
                    for i in range(2, len(acf_values) - 1):
                        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1] and acf_values[i] > 0.3:
                            peaks.append((i, acf_values[i]))
                    
                    if peaks:
                        # Sort by correlation strength
                        peaks.sort(key=lambda x: x[1], reverse=True)
                        detected_seasonality = peaks[0][0]
                        self.logger.info(f"Detected seasonality: {detected_seasonality} periods")
                except ImportError:
                    self.logger.warning("statsmodels not available for seasonality detection")
                except Exception as e:
                    self.logger.warning(f"Error in seasonality detection: {e}")
            
            # Determine which methods to use
            methods_to_use = []
            
            if method == 'auto':
                # Use all available methods
                methods_to_use = ['basic', 'arima', 'prophet']
            else:
                methods_to_use = [method]
            
            forecasts = {}  # type: ignore
            confidence_intervals = {}  # type: ignore
            
            # Always include basic forecasts
            if 'basic' in methods_to_use or method == 'auto':
                basic_forecast = self._implement_basic_forecast(series, forecast_horizon)
                forecasts['basic'] = basic_forecast
            
            # Try ARIMA if available
            if ('arima' in methods_to_use or method == 'auto') and len(series) >= 10:
                try:
                    import statsmodels.api as sm
                    
                    # Prepare data for ARIMA
                    # Convert to stationary if needed
                    diff_order = 0
                    stationary_series = series.copy()
                    
                    # Simple differencing if trend is detected
                    if len(series) > 10:
                        # Check if series is trending using Mann-Kendall test
                        try:
                            from scipy.stats import kendalltau
                            tau, p_value = kendalltau(range(len(series)), series)
                            if p_value < 0.05:  # Significant trend
                                stationary_series = series.diff().dropna()
                                diff_order = 1
                                self.logger.info(f"Series differenced for ARIMA (p-value: {p_value:.4f})")
                        except ImportError:
                            # Simple slope check if scipy.stats not available
                            slope = np.polyfit(range(len(series)), series, 1)[0]
                            if abs(slope) > 0.01 * np.mean(series):
                                stationary_series = series.diff().dropna()
                                diff_order = 1
                                self.logger.info(f"Series differenced for ARIMA (slope: {slope:.4f})")
                    
                    # Determine ARIMA order
                    p, d, q = 1, diff_order, 1  # Default order
                    
                    # Add seasonal component if detected
                    if detected_seasonality > 0:
                        # SARIMA model: (p,d,q) x (P,D,Q,s)
                        seasonal_order = (1, 0, 1, detected_seasonality)
                        model = sm.tsa.SARIMAX(series, order=(p, d, q), 
                                              seasonal_order=seasonal_order,
                                              enforce_stationarity=False)
                        self.logger.info(f"Using SARIMA({p},{d},{q})x{seasonal_order}")
                    else:
                        model = sm.tsa.ARIMA(series, order=(p, d, q))
                        self.logger.info(f"Using ARIMA({p},{d},{q})")
                    
                    # Fit model
                    arima_model = model.fit(disp=0)
                    
                    # Generate forecast
                    arima_forecast = arima_model.forecast(steps=forecast_horizon)
                    
                    # Get confidence intervals
                    if hasattr(arima_model, 'get_forecast'):
                        forecast_obj = arima_model.get_forecast(steps=forecast_horizon)
                        conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% confidence interval
                        confidence_intervals['arima'] = {
                            'lower': conf_int.iloc[:, 0].tolist(),
                            'upper': conf_int.iloc[:, 1].tolist()
                        }
                    
                    forecasts['arima'] = arima_forecast.tolist()
                    self.logger.info("ARIMA forecast completed successfully")
                    
                except ImportError:
                    self.logger.warning("statsmodels not available for ARIMA forecasting")
                except Exception as e:
                    self.logger.error(f"Error in ARIMA forecasting: {e}", exc_info=True)
                    forecasts['arima'] = {"error": str(e)}
            
            # Try Prophet if available
            if ('prophet' in methods_to_use or method == 'auto') and len(series) >= 20:
                try:
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    prophet_df = pd.DataFrame({
                        'ds': df.index if isinstance(df.index, pd.DatetimeIndex) else 
                              pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
                        'y': series.values
                    })
                    
                    # Initialize and fit model
                    prophet_model = Prophet(
                        yearly_seasonality='auto',
                        weekly_seasonality='auto',
                        daily_seasonality=False,
                        seasonality_mode='multiplicative' if np.mean(series) > 0 else 'additive'
                    )
                    
                    # Add custom seasonality if detected
                    if detected_seasonality > 0 and detected_seasonality not in [7, 365]:
                        prophet_model.add_seasonality(
                            name=f'custom_{detected_seasonality}',
                            period=detected_seasonality,
                            fourier_order=min(5, detected_seasonality // 2)
                        )
                    
                    prophet_model.fit(prophet_df)
                    
                    # Create future dataframe
                    future = prophet_model.make_future_dataframe(periods=forecast_horizon, freq='D')
                    
                    # Generate forecast
                    prophet_forecast = prophet_model.predict(future)
                    
                    # Extract forecast values and intervals
                    forecasts['prophet'] = prophet_forecast.tail(forecast_horizon)['yhat'].tolist()
                    confidence_intervals['prophet'] = {
                        'lower': prophet_forecast.tail(forecast_horizon)['yhat_lower'].tolist(),
                        'upper': prophet_forecast.tail(forecast_horizon)['yhat_upper'].tolist()
                    }
                    
                    self.logger.info("Prophet forecast completed successfully")
                    
                except ImportError:
                    self.logger.warning("Prophet not available for forecasting")
                except Exception as e:
                    self.logger.error(f"Error in Prophet forecasting: {e}", exc_info=True)
                    forecasts['prophet'] = {"error": str(e)}
            
            # Use GluonTS DeepAR model if available
            if HAVE_GLUONTS and ('deepar' in methods_to_use or method == 'auto' or method == 'deepar'):
                try:
                    self.logger.info("Using GluonTS DeepAR for advanced time series forecasting")
                    
                    # Convert to proper format for GluonTS
                    gluon_start_time = time.time()
                    freq = "D"  # Default to daily frequency
                    
                    # Convert dataframe index to datetime if needed
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq=freq)
                    else:
                        # Infer frequency from the DatetimeIndex
                        inferred_freq = pd.infer_freq(df.index)
                        if inferred_freq:
                            freq = inferred_freq
                    
                    # Prepare training data for GluonTS
                    training_data = [{
                        "target": series.values,
                        "start": df.index[0],
                        "item_id": column
                    }]
                    
                    # Create GluonTS dataset
                    gluonts_dataset = ListDataset(training_data, freq=freq)
                    
                    # Check if we have a pre-trained model
                    if self.forecast_model and isinstance(self.forecast_model, DeepAREstimator):
                        predictor = self.forecast_model.train(gluonts_dataset)
                        self.logger.info("Using pre-configured DeepAR estimator")
                    else:
                        # Initialize DeepAR estimator with reasonable defaults
                        trainer = Trainer(
                            epochs=10,
                            learning_rate=1e-3,
                            batch_size=32,
                            num_batches_per_epoch=100
                        )
                        
                        estimator = DeepAREstimator(
                            freq=freq,
                            prediction_length=forecast_horizon,
                            trainer=trainer,
                            context_length=min(2 * forecast_horizon, len(series)),
                            num_layers=2,
                            num_cells=40,
                            dropout_rate=0.1
                        )
                        
                        # Train the model
                        self.logger.info("Training DeepAR model")
                        predictor = estimator.train(gluonts_dataset)
                    
                    # Generate forecasts
                    self.logger.info("Generating DeepAR forecasts")
                    forecast_it, ts_it = make_evaluation_predictions(
                        dataset=gluonts_dataset,
                        predictor=predictor,
                        num_samples=100
                    )
                    
                    # Get forecast
                    forecasts_list = list(forecast_it)
                    if forecasts_list:
                        deepar_forecast = forecasts_list[0]
                        # Extract mean forecast
                        mean_forecast = deepar_forecast.mean.tolist()
                        # Extract prediction intervals
                        prediction_intervals = deepar_forecast.quantile(0.05), deepar_forecast.quantile(0.95)
                        
                        # Store results
                        forecasts['deepar'] = mean_forecast
                        confidence_intervals['deepar'] = {
                            'lower': prediction_intervals[0].tolist(),
                            'upper': prediction_intervals[1].tolist()
                        }
                        
                        # Log performance metrics
                        gluon_time = (time.time() - gluon_start_time) * 1000  # ms
                        self.logger.timing("deepar_forecasting_time_ms", gluon_time)
                        self.logger.info(f"DeepAR forecast completed in {gluon_time:.2f}ms")
                    else:
                        self.logger.warning("DeepAR forecasting returned empty results")
                        forecasts['deepar'] = {"error": "Empty forecast results"}
                    
                except Exception as e:
                    self.logger.error(f"Error in DeepAR forecasting: {e}", exc_info=True)
                    forecasts['deepar'] = {"error": str(e)}
            
            # If other advanced model is loaded, use it
            if self.forecast_model and not isinstance(self.forecast_model, DeepAREstimator):
                try:
                    # Handle other types of forecast models
                    model_type = type(self.forecast_model).__name__
                    self.logger.info(f"Using advanced forecast model of type {model_type}")
                    
                    # Handle TensorFlow/Keras models
                    if HAVE_TF and hasattr(self.forecast_model, 'predict'):
                        # Prepare data for TF model - typically needs reshaping
                        # Example for LSTM: input shape [samples, time steps, features]
                        look_back = min(30, len(series) - 1)  # Use last 30 points or less
                        # Convert data to proper format
                        X = series.values[-look_back:].reshape(1, look_back, 1)
                        
                        # Make prediction
                        tf_forecast = []
                        current_input = X
                        
                        for i in range(forecast_horizon):
                            # Get next prediction
                            next_pred = self.forecast_model.predict(current_input)
                            tf_forecast.append(float(next_pred[0, 0]))
                            
                            # Update input for next prediction (rolling window)
                            current_input = np.roll(current_input, -1, axis=1)
                            current_input[0, -1, 0] = next_pred[0, 0]
                        
                        forecasts['tensorflow'] = tf_forecast
                        self.logger.info(f"TensorFlow model forecast generated for {forecast_horizon} steps")
                    
                    # Handle statsmodels models
                    elif HAVE_STATSMODELS and hasattr(self.forecast_model, 'forecast'):
                        sm_forecast = self.forecast_model.forecast(steps=forecast_horizon)
                        forecasts['statsmodel'] = sm_forecast.tolist()
                        self.logger.info(f"Statsmodels forecast generated for {forecast_horizon} steps")
                    
                    # Handle pickle-loaded models with predict method
                    elif hasattr(self.forecast_model, 'predict'):
                        try:
                            # Generic prediction call
                            advanced_forecast = self.forecast_model.predict(df[[column]], horizon=forecast_horizon)
                            forecasts['advanced'] = advanced_forecast.tolist()
                            self.logger.info(f"Custom model forecast generated for {forecast_horizon} steps")
                        except Exception as predict_e:
                            self.logger.error(f"Error calling predict method: {predict_e}", exc_info=True)
                            forecasts['advanced'] = {"error": str(predict_e)}
                    
                    else:
                        self.logger.warning(f"Loaded model of type {model_type} doesn't have standard predict interface")
                        forecasts['advanced'] = {"error": "Model has no standard prediction interface"}
                        
                except Exception as e:
                    self.logger.error(f"Error in advanced model forecasting: {e}", exc_info=True)
                    forecasts['advanced'] = {"error": str(e)}
            
            # Determine best forecast method
            recommended_method = 'basic'
            if 'arima' in forecasts and not isinstance(forecasts['arima'], dict):
                recommended_method = 'arima'
            if 'prophet' in forecasts and not isinstance(forecasts['prophet'], dict):
                recommended_method = 'prophet'
            if 'advanced' in forecasts and not isinstance(forecasts['advanced'], dict):
                recommended_method = 'advanced'
            
            # For basic forecasts, use the recommended method from basic implementation
            if 'basic' in forecasts:
                basic_recommended = forecasts['basic'].get('recommended', 'naive')
                forecasts['basic_recommended'] = forecasts['basic']['values'][basic_recommended]
            
            # Create ensemble forecast (average of all valid forecasts)
            valid_forecasts = []
            for method, forecast in forecasts.items():
                if method != 'basic' and not isinstance(forecast, dict) and method != 'basic_recommended':
                    valid_forecasts.append(forecast)
            
            if 'basic' in forecasts:
                for method, values in forecasts['basic']['values'].items():
                    valid_forecasts.append(values)
            
            if valid_forecasts:
                ensemble = np.mean(valid_forecasts, axis=0).tolist()
                forecasts['ensemble'] = ensemble
                
                # If we have multiple good methods, recommend ensemble
                if len(valid_forecasts) >= 2:
                    recommended_method = 'ensemble'
            
            processing_time = time.time() - start_time
            self.logger.timing("forecast_series_time_ms", processing_time * 1000, 
                              horizon=forecast_horizon, method=method)
            
            return {
                "forecasts": forecasts,
                "confidence_intervals": confidence_intervals,
                "recommended_method": recommended_method,
                "seasonality": detected_seasonality,
                "processing_time": processing_time,
                "column": column
            }

        except Exception as e:
            self.logger.error(f"Error forecasting series: {e}", exc_info=True)
            return {"error": str(e)}
            
    def _implement_basic_forecast(self, series: "pd.Series", horizon: int) -> Dict[str, Any]:
        """
        Implement basic forecasting methods.
        
        Args:
            series: Time series data to forecast
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary with forecast values and metadata
        """
        # Method 1: Naive forecast (last value)
        last_value = float(series.iloc[-1])
        naive_forecast = [last_value] * horizon
        
        # Method 2: Simple Moving Average
        sma_window = min(10, len(series) // 2)  # Use half the series length up to 10
        if sma_window > 0:
            sma = float(series.iloc[-sma_window:].mean())
            sma_forecast = [sma] * horizon
        else:
            sma_forecast = naive_forecast
            
        # Method 3: Linear trend extrapolation
        if len(series) >= 5:
            # Use last 5 points to estimate trend
            y = series.iloc[-5:].values
            x = np.arange(5)
            slope, intercept = np.polyfit(x, y, 1)
            trend_forecast = [float(intercept + slope * (5 + i)) for i in range(horizon)]
        else:
            trend_forecast = naive_forecast
            
        # Method 4: Exponential Smoothing (simple implementation)
        if len(series) >= 3:
            alpha = 0.3  # Smoothing factor
            smoothed = series.iloc[-1]
            exp_forecast = []
            for i in range(horizon):
                if i == 0:
                    # First forecast is based on the last actual value
                    forecast_value = alpha * series.iloc[-1] + (1 - alpha) * smoothed
                else:
                    # Subsequent forecasts are based on previous forecasts
                    forecast_value = alpha * exp_forecast[-1] + (1 - alpha) * smoothed
                exp_forecast.append(float(forecast_value))
        else:
            exp_forecast = naive_forecast
        
        return {
            "values": {
                "naive": naive_forecast,
                "sma": sma_forecast,
                "trend": trend_forecast,
                "exponential": exp_forecast
            },
            "methods": ["naive_last_value", "simple_moving_average", "linear_trend", "exponential_smoothing"],
            "recommended": "trend" if len(series) >= 5 else "naive"
        }
