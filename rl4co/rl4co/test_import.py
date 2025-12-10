import os
import sys
from pathlib import Path


def test_imports():
    """Test different import methods for the ACO baseline."""
    print("Testing imports for ACO baseline...")

    try:
        # Method 1: Direct import from aco module
        print("\nTesting: from baselines.aco import ACOBaseline")
        from baselines.aco import ACOBaseline
        print("✓ Direct import successful")

        # Method 2: Import from baselines package
        print("\nTesting: from baselines import ACOBaseline")
        from baselines import ACOBaseline
        print("✓ Package import successful")

        # Test creating an instance
        print("\nTesting instance creation...")
        from rl4co.envs import TDTSPEnv
        env = TDTSPEnv(matrix_filename="~/ACO/data.csv",
                       generator_params={"num_loc": 20})  # Small test environment
        aco = ACOBaseline(env)
        print("✓ Successfully created ACO instance")

        print("\nAll import tests passed successfully!")
        return True

    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure your project structure is correct:")
        print("   your_project/")
        print("   ├── baselines/")
        print("   │   ├── __init__.py")
        print("   │   └── aco.py")
        print("2. Make sure you're running the script from the correct directory")
        print("3. Check if RL4CO is installed correctly")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    # Add project root to Python path if needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"Added {project_root} to Python path")

    # Run tests
    success = test_imports()
    print(f"\nTest {'passed' if success else 'failed'}")
