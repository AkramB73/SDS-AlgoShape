import pytest
from sds_library.shapes import Circle, Rectangle, Triangle

# A dummy palette is needed to initialize the shapes.
DUMMY_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def test_circle_contains():
    """Tests the point-in-shape logic for the Circle class."""
    circle = Circle(palette=DUMMY_PALETTE)
    # Define a circle centered at (100, 100) with a radius of 50
    circle.center = (100, 100)
    circle.radius = 50

    # Test points that should be inside
    assert circle.contains(100, 100)  # Center point
    assert circle.contains(120, 120)  # A point within the radius

    # Test points that should be outside
    assert not circle.contains(200, 200)  # Far outside
    assert not circle.contains(150, 100)  # Exactly on the edge (should be false)
    assert not circle.contains(50, 100)   # Exactly on the opposite edge

def test_rectangle_contains():
    """Tests the point-in-shape logic for the Rectangle class."""
    rect = Rectangle(palette=DUMMY_PALETTE)
    # Define a 100x80 rectangle starting at (50, 50)
    rect.top_left = (50, 50)
    rect.size = (100, 80)  # Ends at x=150, y=130

    # Test points that should be inside
    assert rect.contains(100, 100) # Center
    assert rect.contains(51, 51)   # Near the top-left corner

    # Test points that should be outside
    assert not rect.contains(49, 100)   # Just outside the left edge
    assert not rect.contains(150, 100)  # Exactly on the right edge (exclusive)
    assert not rect.contains(100, 130)  # Exactly on the bottom edge (exclusive)
    assert not rect.contains(200, 200)  # Far outside

def test_triangle_contains():
    """Tests the point-in-shape logic for the Triangle class."""
    triangle = Triangle(palette=DUMMY_PALETTE)
    # Define a simple right-angled triangle
    triangle.p1 = (50, 50)
    triangle.p2 = (150, 50)
    triangle.p3 = (50, 150)

    # Test points that should be inside
    assert triangle.contains(60, 60)    # Deep inside
    assert triangle.contains(51, 51)    # Near a vertex

    # Test points on the vertices and edges (should be considered inside)
    assert triangle.contains(50, 50)
    assert triangle.contains(100, 50)

    # Test points that should be outside
    assert not triangle.contains(49, 49)    # Just outside
    assert not triangle.contains(151, 51)   # Outside hypotenuse