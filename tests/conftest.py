import pytest
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def setup_warp():
    """Initialize warp once for the entire test session"""
    wp.init()
    # wp.clear_kernel_cache()
    yield