import pytest
import numpy as np
from sds_library.evaluator import partial_test, full_fitness
from sds_library.agent import Agent
from sds_library.shapes import Rectangle

DUMMY_PALETTE = [(0, 0, 0)]  # Black palette 
BACKGROUND_COLOR = (255, 255, 255, 255)  # White canvas
IMG_SIZE = (100, 100)

@pytest.fixture
def simple_target_image():
    """Creates a 100x100 white image with a 50x50 black square in the middle"""
    target = np.full((100, 100, 4), BACKGROUND_COLOR, dtype=np.uint8)
    target[25:75, 25:75] = (0, 0, 0, 255) # Opaque black square
    return target


def test_partial_test_returns_correct_scores(simple_target_image):
    # Create a good agent that perfectly matches the target square
    good_agent = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    good_agent.shapes[0] = Rectangle(DUMMY_PALETTE)
    good_agent.shapes[0].top_left = (25, 25)
    good_agent.shapes[0].size = (50, 50)
    good_agent.shapes[0]._color = (0, 0, 0, 255)

    # Create a bad agent with a shape in the wrong place
    bad_agent = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    bad_agent.shapes[0] = Rectangle(DUMMY_PALETTE)
    bad_agent.shapes[0].top_left = (0, 0)
    bad_agent.shapes[0].size = (10, 10)
    bad_agent.shapes[0]._color = (0, 0, 0, 255)

    samples = [(30, 30), (50, 50), (70, 70)]
    
    good_agent_score = partial_test(good_agent, simple_target_image, samples)
    bad_agent_score = partial_test(bad_agent, simple_target_image, samples)

    assert good_agent_score == 0.0
    assert bad_agent_score > 700.0


def test_full_fitness_improves(simple_target_image):
    # Validate that full_fitness decreases when an agent is improved
    OPAQUE_BLACK = (0, 0, 0, 255)

    # Agent 1: A shape that is slightly off target
    agent1 = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    agent1.shapes[0] = Rectangle(DUMMY_PALETTE)
    agent1.shapes[0].top_left = (20, 20)
    agent1.shapes[0].size = (50, 50)
    agent1.shapes[0]._color = OPAQUE_BLACK

    # Agent 2: A shape that is perfectly on target
    agent2 = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    agent2.shapes[0] = Rectangle(DUMMY_PALETTE)
    agent2.shapes[0].top_left = (25, 25)
    agent2.shapes[0].size = (50, 50)
    agent2.shapes[0]._color = OPAQUE_BLACK

    error1 = full_fitness(agent1, simple_target_image, BACKGROUND_COLOR)
    error2 = full_fitness(agent2, simple_target_image, BACKGROUND_COLOR)

    assert error2 < error1
    assert error2 == 0.0