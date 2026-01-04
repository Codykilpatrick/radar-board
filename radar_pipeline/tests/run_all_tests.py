#!/usr/bin/env python3
"""
Run all radar pipeline tests.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from radar_pipeline.tests.test_kalman import run_all_tests as run_kalman_tests
from radar_pipeline.tests.test_association import run_all_tests as run_association_tests
from radar_pipeline.tests.test_tracking import run_all_tests as run_tracking_tests
from radar_pipeline.tests.test_simulation import run_all_tests as run_simulation_tests


def main():
    print("\n" + "="*60)
    print("RADAR PIPELINE TEST SUITE")
    print("="*60)
    
    try:
        run_kalman_tests()
        run_association_tests()
        run_tracking_tests()
        run_simulation_tests()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

