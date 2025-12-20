"""
Minimal test script to isolate SAC agent initialization issue.

This script tests each step of SAC agent initialization separately to identify
where segmentation faults might occur.
"""
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("SAC Agent Initialization Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"Device available: {torch.cuda.is_available()}")
print("=" * 60)

def test_step(step_name, test_func):
    """Run a test step and handle errors."""
    print(f"\n[Test] {step_name}...")
    sys.stdout.flush()
    try:
        result = test_func()
        print(f"  ✓ {step_name} passed")
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"  ✗ {step_name} failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 1: Basic network creation
def test_basic_network():
    net = nn.Sequential(
        nn.Linear(2, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    return net

test_step("Basic network creation", test_basic_network)

# Test 2: Create MLP function
def test_mlp_function():
    def mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    return mlp

mlp_func = test_step("MLP function definition", test_mlp_function)

# Test 3: Create multiple networks
def test_multiple_networks():
    actor = mlp_func(2, 2)
    q1 = mlp_func(4, 1)
    q2 = mlp_func(4, 1)
    q1_target = mlp_func(4, 1)
    q2_target = mlp_func(4, 1)
    return actor, q1, q2, q1_target, q2_target

actor, q1, q2, q1_target, q2_target = test_step("Multiple networks creation", test_multiple_networks)

# Test 4: Initialize target networks with load_state_dict
def test_load_state_dict():
    try:
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())
        return True
    except Exception as e:
        print(f"  ⚠ load_state_dict failed (this might be the issue): {e}")
        # Try alternative method
        print("  Trying alternative: direct parameter copy...")
        with torch.no_grad():
            for target_param, param in zip(q1_target.parameters(), q1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(q2_target.parameters(), q2.parameters()):
                target_param.data.copy_(param.data)
        return True

test_step("Target network initialization", test_load_state_dict)

# Test 5: Create optimizers
def test_optimizer_creation():
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=3e-4
    )
    return opt_actor, opt_critic

opt_actor, opt_critic = test_step("Optimizer creation", test_optimizer_creation)

# Test 6: Device setup
def test_device_setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device = test_step("Device setup", test_device_setup)

# Test 7: Move networks to device
def test_move_to_device():
    actor.to(device)
    q1.to(device)
    q2.to(device)
    q1_target.to(device)
    q2_target.to(device)
    return True

test_step("Move networks to device", test_move_to_device)

# Test 8: Test actual SACAgent import and creation
def test_sac_agent_import():
    from fedguide.baselines.sac.agent import SACAgent
    return SACAgent

SACAgent = test_step("SACAgent import", test_sac_agent_import)

# Test 9: Create full SACAgent instance
def test_sac_agent_creation():
    agent = SACAgent(
        state_dim=2,
        action_dim=2,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device="cpu",
    )
    return agent

agent = test_step("Full SACAgent creation", test_sac_agent_creation)

# Test 10: Test agent methods
def test_agent_methods():
    # Test act method
    state = torch.randn(1, 2)
    action, log_prob = agent.act(state, eval=False)
    assert action.shape == (1, 2), f"Action shape wrong: {action.shape}"
    
    # Test act with eval=True
    action_eval, _ = agent.act(state, eval=True)
    assert action_eval.shape == (1, 2), f"Eval action shape wrong: {action_eval.shape}"
    
    return True

test_step("Agent methods (act)", test_agent_methods)

# Test 11: Test update method (requires batch)
def test_update_method():
    batch = {
        's': torch.randn(32, 2),
        'a': torch.randn(32, 2),
        'r': torch.randn(32),
        's_next': torch.randn(32, 2),
        'done': torch.zeros(32),
    }
    actor_loss, critic_loss, q_values = agent.update(batch)
    assert isinstance(actor_loss, float), "Actor loss should be float"
    assert isinstance(critic_loss, float), "Critic loss should be float"
    assert q_values.shape == (32,), f"Q values shape wrong: {q_values.shape}"
    return True

test_step("Agent update method", test_update_method)

print("\n" + "=" * 60)
print("✓ All tests passed! SAC agent initialization works correctly.")
print("=" * 60)
print("\nIf you still get segmentation fault when running the full script,")
print("the issue might be:")
print("  1. PyTorch version compatibility")
print("  2. Environment-specific issues (macOS, specific Python version)")
print("  3. Memory issues with large batch sizes")
print("  4. Issues in other parts of the code (data loading, trainer, etc.)")
print("=" * 60)

