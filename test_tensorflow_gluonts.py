#!/usr/bin/env python3
"""
Test script to verify that TensorFlow and GluonTS imports are working correctly.
"""

def test_imports():
    """Test importing TensorFlow and GluonTS modules."""
    try:
        import tensorflow
        print("✅ Successfully imported tensorflow", tensorflow.__version__)
    except ImportError as e:
        print(f"❌ Error importing tensorflow: {e}")
    
    try:
        import tensorflow.keras.models #type: ignore
        import tensorflow.keras.layers #type: ignore
        import tensorflow.keras.callbacks #type: ignore
    except ImportError as e:
        print(f"❌ Error importing tensorflow.keras modules: {e}")
    
    try:
        import gluonts
        print("✅ Successfully imported gluonts", gluonts.__version__)
    except ImportError as e:
        print(f"❌ Error importing gluonts: {e}")
    
    try:
        import gluonts.torch
        print("✅ Successfully imported gluonts.torch")
        
        # Specifically check for DeepAREstimator
        from gluonts.torch import DeepAREstimator
        print("✅ Successfully imported gluonts.torch.DeepAREstimator")
    except ImportError as e:
        print(f"❌ Error importing gluonts.torch: {e}")

if __name__ == "__main__":
    test_imports()
