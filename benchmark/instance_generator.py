import numpy as np
import random as rd

# Instance generator for Online Bin Packing 1D Problem
# Using:
#   - Capacity of 100 for containers
#   - N = 100 items
#   - Uniform and Weibull distribution for item generation
# Weibull distribution a parameter set to 1'5 for Bin Packing Problem (More little and medium packages rather than big ones)
# Weibull distribution scale x45 for container's capacity C=100
# Weibull distribution formula: f(x) = ax^(a-1)exp(-x^a)


def generate_uniform_items(n_items: int = 100, capacity: int = 100, low: int = 10, high: int = 80) -> list[int]:
    """
    Generates a list of n_items integer items with a normmal distribution in the range [low,high] 
    without exceeding container's capacity
    """

    # Generate items with uniform distribution
    items = np.random.randint(low, high + 1, n_items)

    # Make sure they are at least 1 and are not bigger than the container
    items = np.clip(items, 1, capacity - 1)

    return items.tolist()


def generate_weibull_items(n_items: int = 100, capacity: int = 100, a: float = 1.5, scale: int = 45) -> list[int]:
    """
    Generates a list of n_items integer items with a weibull distribution
    without exceeding container's capacity
    """

    # Generates items with a weibull distribution
    raw_items = np.random.weibull(a, n_items)

    # Scale them to accord the bin capacity
    scaled_items = raw_items * scale

    # Round them to make sure they are integer
    items = np.round(scaled_items).astype(int)

    # Make sure they are at least 1 and are not bigger than the container
    items = np.clip(items, 1, capacity - 1)

    return items.tolist()


def create_instance(id:str, items:list[int], capacity: int):
    """
    Returns an instance with the format:
    - instance_id: The id for that instance, composed of the distribution used plus a number
    - bin_capacity: The capacity of the containers
    - items: The list of items shuffled
    - lower_bound: The theorical lower bound 
    """

    # Shuffle
    rd.shuffle(items)

    # Calculate lower_bound
    lower_bound = int(np.ceil(sum(items) / capacity))

    return {
        "instance_id" : id,
        "bin_capacity" : capacity,
        "items" : items,
        "lower_bound" : lower_bound
    }


def generate_benchmark_batch(n_instances: int = 10, n_items: int = 100, capacity: int = 100, seed: int = 42) -> dict:
    """
    Generates a benchmark batch
    """
    # Set seed
    rd.seed(seed)
    np.random.seed(seed)

    # Create instances list
    uniform_instances = []
    weibull_instances = []

    for i in range(n_instances):
        # Uniform instance
        u_items = generate_uniform_items(n_items, capacity)
        uniform_instances.append(create_instance(f"uniform_instance_{i:02d}", u_items, capacity))

        # Weibull instance
        w_items = generate_weibull_items(n_items, capacity)
        weibull_instances.append(create_instance(f"weibull_instance_{i:02d}", w_items, capacity))

    return {
        "uniform" : uniform_instances,
        "weibull" : weibull_instances
    }


if __name__ == "__main__":
    batch = generate_benchmark_batch()

    print(batch)