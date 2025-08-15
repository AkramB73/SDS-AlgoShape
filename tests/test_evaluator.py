import pytest
import numpy as np
from sds_library.evaluator import partial_test, full_fitness
from sds_library.agent import Agent
from sds_library.shapes import Rectangle

DUMMY_PALETTE = [(255, 255, 255)]  # A white palette
BACKGROUND_COLOR = (0, 0, 0, 255) # Black background
IMG_SIZE = (100, 100)

@pytest.fixture
def simple_target_image():
    """Creates a 100x100 black image with a 50x50 white square in the middle."""
    target = np.full((100, 100, 4), BACKGROUND_COLOR, dtype=np.uint8)
    target[25:75, 25:75] = (255, 255, 255, 255) # White square
    return target

# --- Tests ---

def test_partial_test_logic(simple_target_image):
    # Create a good agent with one shape that perfectly matches the target
    good_agent = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    good_agent.shapes[0] = Rectangle(DUMMY_PALETTE)
    good_agent.shapes[0].top_left = (25, 25)
    good_agent.shapes[0].size = (50, 50)
    good_agent.shapes[0]._color = (255, 255, 255, 255) # Make it opaque white

    # Create a bad agent with a shape in the wrong place
    bad_agent = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    bad_agent.shapes[0] = Rectangle(DUMMY_PALETTE)
    bad_agent.shapes[0].top_left = (0, 0)
    bad_agent.shapes[0].size = (10, 10)

    # Sample points inside the white square where the good agent matches
    samples = [(30, 30), (50, 50), (70, 70)]
    
    assert partial_test(good_agent, simple_target_image, samples, BACKGROUND_COLOR, error_threshold=50.0)

    assert not partial_test(bad_agent, simple_target_image, samples, BACKGROUND_COLOR, error_threshold=50.0)

def test_full_fitness_improves(simple_target_image):
    """
    Validate that full_fitness decreases when an agent is improved.
    """
    # Agent 1: A shape that is slightly off-target
    agent1 = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    agent1.shapes[0] = Rectangle(DUMMY_PALETTE)
    agent1.shapes[0].top_left = (20, 20) # Slightly offset
    agent1.shapes[0].size = (50, 50)
    
    # Agent 2: A shape that is perfectly on-target
    agent2 = Agent(IMG_SIZE, DUMMY_PALETTE, 1)
    agent2.shapes[0] = Rectangle(DUMMY_PALETTE)
    agent2.shapes[0].top_left = (25, 25) # Perfectly aligned
    agent2.shapes[0].size = (50, 50)

    error1 = full_fitness(agent1, simple_target_image, BACKGROUND_COLOR)
    error2 = full_fitness(agent2, simple_target_image, BACKGROUND_COLOR)

    # The error for the perfectly aligned agent should be lower than the offset one
    assert error2 < error1