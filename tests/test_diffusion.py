import pytest
import numpy as np
import copy
import uuid
from sds_library.diffusion import DiffusionSearch
from sds_library.agent import Agent
from sds_library.shapes import Rectangle

IMG_SIZE = (100, 100)
BACKGROUND_COLOR = (255, 255, 255, 255)
PALETTE = [(0, 0, 0)]
N_AGENTS = 3

@pytest.fixture
def simple_target_image():
    """Creates a 100x100 white image with a 50x50 black square"""
    target = np.full((*IMG_SIZE, 4), BACKGROUND_COLOR, dtype=np.uint8)
    target[25:75, 25:75] = (0, 0, 0, 255)
    return target

@pytest.fixture
def sds_instance(simple_target_image):
    return DiffusionSearch(
        target=simple_target_image, palette=PALETTE, n_agents=N_AGENTS,
        shapes_per_agent=1, n_samples=10, start_threshold=255.0, end_threshold=10.0
    )

# Unit Tests

def test_step_logic_with_new_threshold(sds_instance):
    perfect_agent = sds_instance.population[0]
    perfect_shape = Rectangle(PALETTE)
    perfect_shape.top_left = (25, 25)
    perfect_shape.size = (50, 50)
    perfect_shape._color = (0, 0, 0, 255)
    perfect_agent.shapes = [perfect_shape]

    for i in range(1, N_AGENTS):
        bad_agent = sds_instance.population[i]
        bad_shape = Rectangle(PALETTE)
        bad_shape.top_left = (0, 0)
        bad_shape.size = (1, 1)
        bad_shape._color = (0, 0, 0, 255)
        bad_agent.shapes = [bad_shape]

    original_bad_agent_shapes = [
        copy.deepcopy(agent.shapes[0]) for agent in sds_instance.population[1:]
    ]

    sds_instance.step(current_iteration=0, total_iterations=100)

    assert sds_instance.population[0].is_active
    for i in range(1, N_AGENTS):
        agent = sds_instance.population[i]
        assert not agent.is_active
        current_shape = agent.shapes[0]
        original_shape = original_bad_agent_shapes[i - 1]
        
        if isinstance(current_shape, Rectangle):
            assert (current_shape.top_left != original_shape.top_left or
                    current_shape.size != original_shape.size)
        else:
            assert not isinstance(current_shape, type(original_shape))

def test_get_best_agents_sorts_by_real_fitness(sds_instance):

    population = sds_instance.population
        
    # Perfect score 
    perfect_agent = population[1]
    perfect_agent.shapes[0] = Rectangle(PALETTE)
    perfect_agent.shapes[0].top_left = (25, 25)
    perfect_agent.shapes[0].size = (50, 50)
    perfect_agent.shapes[0]._color = (0, 0, 0, 255)

    # Small error score
    good_agent = population[2]
    good_agent.shapes[0] = Rectangle(PALETTE)
    good_agent.shapes[0].top_left = (26, 25) # off by 1 pixel            
    good_agent.shapes[0].size = (50, 50)
    good_agent.shapes[0]._color = (0, 0, 0, 255)

    # High error score
    bad_agent = population[0]
    bad_agent.shapes[0] = Rectangle(PALETTE)
    bad_agent.shapes[0].top_left = (0, 0)
    bad_agent.shapes[0].size = (1, 1)
    bad_agent.shapes[0]._color = (0, 0, 0, 255)

    top_2_agents = sds_instance.get_best_agents(n=2)

    assert len(top_2_agents) == 2
    assert top_2_agents[0].id == perfect_agent.id
    assert top_2_agents[1].id == good_agent.id