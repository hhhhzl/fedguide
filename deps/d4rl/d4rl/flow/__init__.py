import gym
import os
from d4rl import offline_env
from gym.envs.registration import register

from copy import deepcopy

import flow
import flow.envs
from flow.networks.ring import RingNetwork
from flow.core.params import NetParams, VehicleParams, EnvParams, InFlows
from flow.core.params import SumoLaneChangeParams, SumoCarFollowingParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter 
from flow.controllers import SimCarFollowingController, SimLaneChangeController
from flow.controllers import RLController
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
from flow.envs import WaveAttenuationPOEnv
from flow.envs import BayBridgeEnv, TrafficLightGridPOEnv

from d4rl.flow import traffic_light_grid
from d4rl.flow import merge
from d4rl.flow import bottleneck

def flow_register(flow_params, render=None, **kwargs):
    exp_tag = flow_params["exp_tag"]
    env_params = flow_params['env']
    net_params = flow_params['net']
    env_class = flow_params['env_name']
    initial_config = flow_params.get('initial', InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLightParams())
    sim_params = deepcopy(flow_params['sim'])
    vehicles = deepcopy(flow_params['veh'])

    sim_params.render = render or sim_params.render

    if isinstance(flow_params["network"], str):
        print("""Passing of strings for network will be deprecated.
        Please pass the Network instance instead.""")
        module = __import__("flow.networks", fromlist=[flow_params["network"]])
        network_class = getattr(module, flow_params["network"])
    else:
        network_class = flow_params["network"]

    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    flow_env = env_class(
        env_params= env_params,
        sim_params= sim_params,
        network= network,
        simulator= flow_params['simulator']
    )

    env = offline_env.OfflineEnvWrapper(flow_env,
        **kwargs
    )
    return env


def ring_env(render='drgb'):
    name = "ring"
    network_name = RingNetwork
    env_name = WaveAttenuationPOEnv

    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    initial_config = InitialConfig(spacing="uniform", shuffle=False)

    vehicles = VehicleParams()
    vehicles.add("human",
                 acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=21)
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)

    sim_params = SumoParams(sim_step=0.5, render=render, save_render=True)
    HORIZON=100
    env_params = EnvParams(
        # length of one rollout
        horizon=HORIZON,
        additional_params={
            # maximum acceleration of autonomous vehicles
            "max_accel": 1,
            # maximum deceleration of autonomous vehicles
            "max_decel": 1,
            # bounds on the ranges of ring road lengths the autonomous vehicle 
            # is trained on
            "ring_length": [220, 270],
        },
    )


    flow_params = dict(
        exp_tag=name,
        env_name=env_name,
        network=network_name,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config
    )
    return flow_params


RING_RANDOM_SCORE = -165.22
RING_EXPERT_SCORE = 24.42

register(
    id='flow-ring-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render=False),
        'dataset_url': None,
        'ref_min_score': RING_RANDOM_SCORE,
        'ref_max_score': RING_EXPERT_SCORE
    }
)


register(
    id='flow-ring-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render='drgb'),
        'dataset_url': None,
        'ref_min_score': RING_RANDOM_SCORE,
        'ref_max_score': RING_EXPERT_SCORE
    }
)

register(
    id='flow-ring-random-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render=False),
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-random.hdf5',
        'ref_min_score': RING_RANDOM_SCORE,
        'ref_max_score': RING_EXPERT_SCORE
    }
)


register(
    id='flow-ring-controller-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=500,
    kwargs={
        'flow_params': ring_env(render=False),
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-idm.hdf5',
        'ref_min_score': RING_RANDOM_SCORE,
        'ref_max_score': RING_EXPERT_SCORE
    }
)


MERGE_RANDOM_SCORE = 118.67993
MERGE_EXPERT_SCORE = 330.03179

register(
    id='flow-merge-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=750,
    kwargs={
        'flow_params': merge.gen_env(render=False),
        'dataset_url': None,
        'ref_min_score': MERGE_RANDOM_SCORE,
        'ref_max_score': MERGE_EXPERT_SCORE
    }
)


register(
    id='flow-merge-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=750,
    kwargs={
        'flow_params': merge.gen_env(render='drgb'),
        'dataset_url': None,
        'ref_min_score': MERGE_RANDOM_SCORE,
        'ref_max_score': MERGE_EXPERT_SCORE
    }
)

register(
    id='flow-merge-random-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=750,
    kwargs={
        'flow_params': merge.gen_env(render=False),
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-random.hdf5',
        'ref_min_score': MERGE_RANDOM_SCORE,
        'ref_max_score': MERGE_EXPERT_SCORE
    }
)

register(
    id='flow-merge-controller-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=750,
    kwargs={
        'flow_params': merge.gen_env(render=False),
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-idm.hdf5',
        'ref_min_score': MERGE_RANDOM_SCORE,
        'ref_max_score': MERGE_EXPERT_SCORE
    }
)


# FigureEight environments
def figureeight_env_v1(render='drgb'):
    """FigureEight1 environment configuration from flow.benchmarks.figureeight1"""
    try:
        from flow.benchmarks.figureeight1 import flow_params as flow_params_v1
        # flow_params_v1 should be a dictionary with all necessary parameters
        # If it's already a dict, return it directly
        if isinstance(flow_params_v1, dict):
            # Make a copy and update render settings
            params = deepcopy(flow_params_v1)
            if 'sim' in params and hasattr(params['sim'], 'render'):
                params['sim'].render = render
            elif 'sim' in params:
                params['sim'] = deepcopy(params['sim'])
                params['sim'].render = render
            return params
        # If it's a function or callable, call it
        elif callable(flow_params_v1):
            return flow_params_v1(render=render)
        else:
            return flow_params_v1
    except ImportError as e:
        raise ImportError(
            "flow.benchmarks.figureeight1 not found. "
            "Make sure flow is properly installed with benchmarks."
        ) from e


def figureeight_env_v2(render='drgb'):
    """FigureEight2 environment configuration from flow.benchmarks.figureeight2"""
    try:
        from flow.benchmarks.figureeight2 import flow_params as flow_params_v2
        # flow_params_v2 should be a dictionary with all necessary parameters
        if isinstance(flow_params_v2, dict):
            params = deepcopy(flow_params_v2)
            if 'sim' in params and hasattr(params['sim'], 'render'):
                params['sim'].render = render
            elif 'sim' in params:
                params['sim'] = deepcopy(params['sim'])
                params['sim'].render = render
            return params
        elif callable(flow_params_v2):
            return flow_params_v2(render=render)
        else:
            return flow_params_v2
    except ImportError as e:
        raise ImportError(
            "flow.benchmarks.figureeight2 not found. "
            "Make sure flow is properly installed with benchmarks."
        ) from e


# Reference scores for FigureEight (adjust these based on actual performance)
FIGUREEIGHT1_RANDOM_SCORE = 0.0
FIGUREEIGHT1_EXPERT_SCORE = 1.0
FIGUREEIGHT2_RANDOM_SCORE = 0.0
FIGUREEIGHT2_EXPERT_SCORE = 1.0

# Register FigureEight1 environment
register(
    id='flow-figureeight1-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1500,  # Adjust based on actual horizon
    kwargs={
        'flow_params': figureeight_env_v1(render=False),
        'dataset_url': None,  # Add dataset URL if available
        'ref_min_score': FIGUREEIGHT1_RANDOM_SCORE,
        'ref_max_score': FIGUREEIGHT1_EXPERT_SCORE,
    }
)

# Register FigureEight1 with rendering
register(
    id='flow-figureeight1-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1500,
    kwargs={
        'flow_params': figureeight_env_v1(render='drgb'),
        'dataset_url': None,
        'ref_min_score': FIGUREEIGHT1_RANDOM_SCORE,
        'ref_max_score': FIGUREEIGHT1_EXPERT_SCORE,
    }
)

# Register FigureEight2 environment
register(
    id='flow-figureeight2-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1500,  # Adjust based on actual horizon
    kwargs={
        'flow_params': figureeight_env_v2(render=False),
        'dataset_url': None,  # Add dataset URL if available
        'ref_min_score': FIGUREEIGHT2_RANDOM_SCORE,
        'ref_max_score': FIGUREEIGHT2_EXPERT_SCORE,
    }
)

# Register FigureEight2 with rendering
register(
    id='flow-figureeight2-render-v0',
    entry_point='d4rl.flow:flow_register',
    max_episode_steps=1500,
    kwargs={
        'flow_params': figureeight_env_v2(render='drgb'),
        'dataset_url': None,
        'ref_min_score': FIGUREEIGHT2_RANDOM_SCORE,
        'ref_max_score': FIGUREEIGHT2_EXPERT_SCORE,
    }
)

