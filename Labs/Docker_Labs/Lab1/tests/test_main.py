#!/usr/bin/env python
"""
Test improvements work locally before Docker build
"""

import sys
import os

# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_dir)

print(f"Adding to path: {src_dir}")
print(f"src exists: {os.path.exists(src_dir)}")
print()

def test_imports():
    """Test all imports work"""
    try:
        from main import (
            validate_data, train_model, evaluate_model, 
            save_model_and_metrics, make_sample_predictions, main
        )
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_pipeline():
    """Test the complete pipeline"""
    try:
        # Get absolute path to src directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(os.path.dirname(current_dir), 'src')
        
        # Save original directory
        original_dir = os.getcwd()
        
        # Change to src directory to run main
        os.chdir(src_dir)
        
        # Import and run
        from main import main
        
        print("\n" + "="*60)
        print("Running improved pipeline locally...")
        print("="*60 + "\n")
        
        result = main()
        
        if result == 0:
            print("\n" + "="*60)
            print("LOCAL VERIFICATION SUCCESSFUL")
            print("="*60)
            print(f"\nGenerated files in: {src_dir}")
            print("  - iris_model.pkl")
            print("  - model_metrics.json")
            print("  - sample_predictions.json")
            print("\n Ready to build Docker image!")
            
            # Change back to original directory
            os.chdir(original_dir)
            return True
        else:
            print("\n Pipeline returned error")
            os.chdir(original_dir)
            return False
            
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        # Change back to original directory
        try:
            os.chdir(original_dir)
        except:
            pass
        return False

if __name__ == "__main__":
    print("Testing improved Docker Lab 1 locally...")
    print()
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test pipeline
    if not test_pipeline():
        sys.exit(1)
    
    print("\n All checks passed! You can now build the Docker image.")
    sys.exit(0)