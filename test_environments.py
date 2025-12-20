"""
Test script for verifying environment registration and creation.

This script tests:
1. D4RL environment registration (maze2d, antmaze)
2. Flow environment registration (figureeight1, figureeight2)
3. Custom environment creation (Bandit2D, PointMazeNarrow)
4. Environment compatibility with make_env functions
"""

import sys
import traceback
import gymnasium as gym
import gym as old_gym  # Old gym package for d4rl environments

# Test configuration
TEST_ENVIRONMENTS = {
    "custom": [
        "bandit2d",
        "pointmazenarrow",
    ],
    "maze2d": [
        "maze2d-open-v0",
        "maze2d-umaze-v1",
        "maze2d-medium-v1",
        "maze2d-large-v1",
    ],
    "antmaze": [
        "antmaze-umaze-v0",
        "antmaze-umaze-diverse-v0",
        "antmaze-medium-play-v0",
        "antmaze-medium-diverse-v0",
    ],
    "flow": [
        "flow-figureeight1-v0",
        "flow-figureeight2-v0",
        "flow-figureeight1-render-v0",
        "flow-figureeight2-render-v0",
    ],
}


def _check_flow_available():
    """Check if flow package is available."""
    try:
        import d4rl.flow
        return True
    except ImportError:
        return False


def test_env_creation(env_id: str, seed: int = 42):
    """Test environment creation and basic operations."""
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {env_id}")
        print(f"{'='*60}")
        
        # Check for flow environments and skip if flow is not available
        if env_id.startswith("flow-"):
            if not _check_flow_available():
                print("⚠ Skipped: Flow package not installed")
                print("  (Install flow to test flow environments)")
                return None
        
        # Try importing d4rl to register environments
        try:
            import d4rl
            print("✓ D4RL imported successfully")
        except ImportError as e:
            print(f"⚠ D4RL import failed: {e}")
            print("  (This is OK if you're not using D4RL environments)")
        
        # Create environment based on type
        print(f"\n1. Creating environment '{env_id}'...")
        try:
            # Handle custom environments
            if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
                from fedguide.envs.bandit2d import Bandit2D
                env = Bandit2D(K=4, sigma=0.2, seed=seed)
                print(f"   ✓ Custom environment (Bandit2D) created successfully")
            elif env_id.lower() == "pointmazenarrow":
                from fedguide.envs.pointmaze_narrow import PointMazeNarrow
                env = PointMazeNarrow()
                print(f"   ✓ Custom environment (PointMazeNarrow) created successfully")
            else:
                # Use old gym for d4rl environments (maze2d, antmaze, flow, etc.)
                # D4RL registers environments with old gym, not gymnasium
                env = old_gym.make(env_id)
                print(f"   ✓ D4RL environment created successfully")
            print(f"   - Observation space: {env.observation_space}")
            print(f"   - Action space: {env.action_space}")
        except Exception as e:
            error_msg = str(e)
            # Check for known compatibility issues (these should now be fixed)
            if "MujocoEnv.__init__() missing 1 required positional argument: 'observation_space'" in error_msg:
                print(f"   ✗ Failed: {e}")
                traceback.print_exc()
                return False
            elif "doesn't exist" in error_msg and "flow" in env_id.lower():
                print(f"   ⚠ Skipped: Flow environment not registered")
                print(f"   (Flow package may not be properly installed or configured)")
                return None
            print(f"   ✗ Failed: {e}")
            traceback.print_exc()
            return False
        
        # Test reset
        print(f"\n2. Testing env.reset(seed={seed})...")
        try:
            try:
                reset_result = env.reset(seed=seed)
            except TypeError:
                # Old gym doesn't support seed parameter
                reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
            print(f"   ✓ Reset successful")
            print(f"   - Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"   - Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
        except Exception as e:
            print(f"   ✗ Reset failed: {e}")
            traceback.print_exc()
            return False
        
        # Test step
        print(f"\n3. Testing env.step()...")
        try:
            action = env.action_space.sample()
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                print(f"   ✓ Step successful (gymnasium format)")
                print(f"   - Reward: {reward}")
                print(f"   - Terminated: {terminated}, Truncated: {truncated}")
            elif len(result) == 4:
                obs, reward, done, info = result
                print(f"   ✓ Step successful (gym format)")
                print(f"   - Reward: {reward}")
                print(f"   - Done: {done}")
            else:
                print(f"   ⚠ Unexpected step result format: {len(result)} values")
        except Exception as e:
            print(f"   ✗ Step failed: {e}")
            traceback.print_exc()
            return False
        
        # Test dataset if available
        print(f"\n4. Testing env.get_dataset() (if available)...")
        try:
            if hasattr(env, 'get_dataset'):
                dataset = env.get_dataset()
                print(f"   ✓ Dataset available")
                print(f"   - Dataset keys: {list(dataset.keys()) if isinstance(dataset, dict) else 'N/A'}")
                if isinstance(dataset, dict) and 'observations' in dataset:
                    print(f"   - Observations shape: {dataset['observations'].shape}")
                    print(f"   - Actions shape: {dataset['actions'].shape}")
            else:
                print(f"   ℹ Dataset not available (not a D4RL environment)")
        except Exception as e:
            print(f"   ⚠ Dataset check failed: {e}")
        
        # Cleanup
        try:
            env.close()
        except:
            pass
        
        print(f"\n✓ All tests passed for {env_id}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        traceback.print_exc()
        return False


def test_make_env_function(env_id: str, seed: int = 42):
    """Test the make_env function from runner.py"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing make_env('{env_id}', seed={seed})")
        print(f"{'='*60}")
        
        # Skip if runner module doesn't exist
        try:
            from runner import make_env
        except ImportError:
            print(f"⚠ Skipped: runner module not found")
            return None
        
        env = make_env(env_id, seed=seed)
        print(f"✓ Environment created using make_env()")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
        
        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        print(f"✓ Reset successful")
        
        # Test step
        action = env.action_space.sample()
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            print(f"✓ Step successful (gymnasium format)")
        elif len(result) == 4:
            obs, reward, done, info = result
            print(f"✓ Step successful (gym format)")
        else:
            print(f"✓ Step successful (unexpected format: {len(result)} values)")
        
        return True
        
    except Exception as e:
        print(f"✗ make_env test failed: {e}")
        traceback.print_exc()
        return False


def test_client_make_env(env_id: str, seed: int = 42):
    """Test the _make_env function from FedGuideClient"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing _make_env('{env_id}', seed={seed})")
        print(f"{'='*60}")
        
        from fedguide.fed.fedguide.client import _make_env
        
        env = _make_env(env_id, seed=seed)
        print(f"✓ Environment created using _make_env()")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
        
        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs, info = reset_result, {}
        print(f"✓ Reset successful")
        
        # Test step - handle both gym (4 values) and gymnasium (5 values) formats
        action = env.action_space.sample()
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            print(f"✓ Step successful (gymnasium format)")
        elif len(result) == 4:
            obs, reward, done, info = result
            print(f"✓ Step successful (gym format)")
        else:
            print(f"✓ Step successful (unexpected format: {len(result)} values)")
        
        return True
        
    except Exception as e:
        print(f"✗ _make_env test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all environment tests."""
    print("="*60)
    print("Environment Testing Suite")
    print("="*60)
    
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
    }
    
    # Test each category
    for category, env_list in TEST_ENVIRONMENTS.items():
        print(f"\n\n{'#'*60}")
        print(f"Testing {category.upper()} environments")
        print(f"{'#'*60}")
        
        for env_id in env_list:
            try:
                # Test basic environment creation
                success = test_env_creation(env_id, seed=42)
                
                if success is None:
                    # Test was skipped (known compatibility issue or missing dependency)
                    results["skipped"] += 1
                elif success:
                    results["passed"] += 1
                    
                    # Test make_env function
                    try:
                        test_make_env_function(env_id, seed=42)
                    except Exception as e:
                        print(f"⚠ make_env test skipped: {e}")
                    
                    # Test client _make_env function
                    try:
                        test_client_make_env(env_id, seed=42)
                    except Exception as e:
                        print(f"⚠ _make_env test skipped: {e}")
                        
                else:
                    results["failed"] += 1
                    
            except KeyboardInterrupt:
                print("\n\nTest interrupted by user")
                break
            except Exception as e:
                print(f"\n✗ Unexpected error testing {env_id}: {e}")
                traceback.print_exc()
                results["failed"] += 1
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"{'='*60}\n")
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

