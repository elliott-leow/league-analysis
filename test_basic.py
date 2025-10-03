#!/usr/bin/env python3
"""
Basic test script to verify the project setup.

Run with: python3 test_basic.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src import config
        print("  ✓ config")
    except ImportError as e:
        print(f"  ✗ config: {e}")
        return False
    
    try:
        from src import variables
        print("  ✓ variables")
    except ImportError as e:
        print(f"  ✗ variables: {e}")
        return False
    
    try:
        from src import discretization
        print("  ✓ discretization")
    except ImportError as e:
        print(f"  ✗ discretization: {e}")
        return False
    
    try:
        from src import preprocessing
        print("  ✓ preprocessing")
    except ImportError as e:
        print(f"  ✗ preprocessing: {e}")
        return False
    
    try:
        from src import ges
        print("  ✓ ges")
    except ImportError as e:
        print(f"  ✗ ges: {e}")
        return False
    
    try:
        from src import parameters
        print("  ✓ parameters")
    except ImportError as e:
        print(f"  ✗ parameters: {e}")
        return False
    
    try:
        from src import queries
        print("  ✓ queries")
    except ImportError as e:
        print(f"  ✗ queries: {e}")
        return False
    
    try:
        from src import visualize
        print("  ✓ visualize")
    except ImportError as e:
        print(f"  ✗ visualize: {e}")
        return False
    
    try:
        from src import compare
        print("  ✓ compare")
    except ImportError as e:
        print(f"  ✗ compare: {e}")
        return False
    
    try:
        from src import cli
        print("  ✓ cli")
    except ImportError as e:
        print(f"  ✗ cli: {e}")
        return False
    
    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    from src import config
    
    # Check paths exist
    assert config.PROJECT_ROOT.exists(), "Project root not found"
    print(f"  ✓ Project root: {config.PROJECT_ROOT}")
    
    assert config.DATA_DIR.exists(), "Data directory not found"
    print(f"  ✓ Data directory: {config.DATA_DIR}")
    
    # Check data files exist
    assert config.MATCH_DATA_FILE.exists(), "matchData.csv not found"
    print(f"  ✓ matchData.csv found")
    
    assert config.MATCH_IDS_FILE.exists(), "match_ids.csv not found"
    print(f"  ✓ match_ids.csv found")
    
    # Check variables
    assert len(config.VARIABLES) == 9, f"Expected 9 variables, got {len(config.VARIABLES)}"
    print(f"  ✓ Variables defined: {config.VARIABLES}")
    
    return True


def test_variables():
    """Test variable schema."""
    print("\nTesting variable schema...")
    
    from src import variables
    
    all_vars = variables.get_all_variables()
    assert len(all_vars) == 9, f"Expected 9 variables, got {len(all_vars)}"
    print(f"  ✓ {len(all_vars)} variables defined")
    
    # Test temporal ordering
    temporal_groups = variables.get_variables_by_temporal_order()
    print(f"  ✓ Temporal groups: {list(temporal_groups.keys())}")
    
    # Test edge validity
    assert variables.can_edge_exist("FB", "Win"), "FB -> Win should be valid"
    assert not variables.can_edge_exist("Win", "FB"), "Win -> FB should be invalid"
    print("  ✓ Edge validity checks working")
    
    return True


def test_discretization():
    """Test discretization functions."""
    print("\nTesting discretization...")
    
    import pandas as pd
    import numpy as np
    from src import discretization
    
    # Create test data
    test_data = pd.DataFrame({
        'Gold10': [-2000, -500, 0, 500, 2000],
        'Gold20': [-5000, -2000, 0, 2000, 5000],
        'Drakes': [0, 1, 2, 3, 5],
        'Baron': [0, 0, 1, 1, 2],
        'Towers': [-3, -1, 0, 1, 3],
        'FB': [0, 1, 0, 1, 1],
        'FT': [0, 0, 1, 1, 1],
        'Soul': ['None', 'Infernal', 'None', 'Mountain', 'Ocean'],
        'Win': [0, 0, 1, 1, 1]
    })
    
    # Apply discretization
    discretized = discretization.discretize_all_variables(test_data)
    
    # Check that all values are strings
    for col in discretized.columns:
        assert discretized[col].dtype == 'object', f"{col} should be string/object type"
    
    print("  ✓ Discretization working correctly")
    print(f"  Sample output:\n{discretized.head()}")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("BASIC TEST SUITE")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Variables", test_variables),
        ("Discretization", test_discretization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} test FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! Project is ready to use.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


