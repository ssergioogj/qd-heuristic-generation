# For evaluating fitness in execution time

from eval.get_instance import GetData
import numpy as np


class RuntimeBatchData(GetData):

  def __init__(self, batch_items: int = 250) -> None:
    super().__init__()
    for dataset in self.datasets.values():
      for instance in dataset.values():
        # Use the first N items for reducing runtime
        instance['items'] = np.array(
            instance['items'][:batch_items], dtype=np.int32
        )
        instance['num_items'] = len(instance['items'])