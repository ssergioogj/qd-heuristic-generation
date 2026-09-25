# EoH imlpementation
# Repo: https://github.com/FeiLiu36/EoH

# -------
# Evaluation code for EoH on Online Bin Packing
# --------
# Benchmarks a set of scoring heuristics on the Weibull test datasets with
# 1k / 5k / 10k items, each under bin capacity 100 and 500. Reports the excess
# over the L1 lower bound (lower is better) as a method x configuration table.
#
# More results may refer to
# Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang
# "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model"
# ICML 2024, https://arxiv.org/abs/2401.02051.

import importlib
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BP_ROOT = os.path.join(HERE, '..')
DATA_DIR = os.path.join(BP_ROOT, 'testingdata')

# Make get_instance / evaluation and the local heuristic modules importable.
sys.path.insert(0, BP_ROOT)
sys.path.insert(0, HERE)

from get_instance import GetData
from evaluation import Evaluation


def load_raw(pkl_name):
    """Load the raw item lists from a Weibull test pickle."""
    with open(os.path.join(DATA_DIR, pkl_name), 'rb') as f:
        raw = pickle.load(f)
    item_lists = []
    for _num_items, instances in raw.items():
        item_lists.extend(list(items) for items in instances)
    return item_lists


def build_dataset(item_lists, capacity):
    """Wrap raw item lists into the {instance_name: {...}} format for a capacity."""
    return {
        f'test_{i}': {
            'capacity': capacity,
            'num_items': len(items),
            'items': items,
        }
        for i, items in enumerate(item_lists, 1)
    }


# Heuristics to benchmark: (module name in this directory, display name).
HEURISTICS = [
    ('baselines.heuristic_best_fit', 'Best Fit'),
    ('baselines.heuristic_first_fit', 'First Fit'),
    ('baselines.heuristic_eoc', 'EoC'),
    ('baselines.heuristic_eoh', 'EoH'),
    ('baselines.heuristic_fun_search', 'FunSearch'),
    ('baselines.heuristic_fun_search_best_performing', 'FunSearch Best Performing (OR Dataset)'),
]

# Include the freshly evolved heuristic too, if one has been exported.
if os.path.exists(os.path.join(HERE, 'heuristic.py')):
    HEURISTICS.append(('heuristic', 'EoH (evolved)'))

# Test datasets (size label -> pickle) and bin capacities to sweep.
SIZES = [('1k', 'test_dataset_1k.pkl'),
         ('5k', 'test_dataset_5k.pkl'),
         ('10k', 'test_dataset_10k.pkl')]
CAPACITIES = [100, 500]

# Column order: all sizes at c100, then all sizes at c500.
COLUMNS = [f'{size}_c{cap}' for cap in CAPACITIES for size, _ in SIZES]

# Pre-load raw items once per size.
RAW = {size: load_raw(pkl) for size, pkl in SIZES}

eva = Evaluation()
gd = GetData()

# results[display_name][column] = metric
results_aeb = {disp: {} for _, disp in HEURISTICS}
results_time = {disp: {} for _, disp in HEURISTICS}
results_mem = {disp: {} for _, disp in HEURISTICS}
loaded = []
for mod_name, disp in HEURISTICS:
    try:
        heuristic = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001
        print(f'  [skip] {mod_name}: {e}')
        continue
    loaded.append(disp)
    for cap in CAPACITIES:
        for size, _ in SIZES:
            if mod_name == "baselines.heuristic_fun_search_best_performing": print(f"cap: {cap}, size: {size}")
            col_key = f'{size}_c{cap}'
            dataset = build_dataset(RAW[size], cap)
            metrics = eva.evaluateGreedy(dataset, heuristic)
            avg_num_bins = -metrics["avg_num_bins"]
            avg_exec_time = metrics["avg_time_sec"]
            avg_mem = metrics["avg_peak_mem_kb"]
            lb = gd.l1_bound_dataset(dataset)
            aeb = (avg_num_bins - lb) / lb
            results_aeb[disp][col_key] = aeb
            results_time[disp][col_key] = avg_exec_time
            results_mem[disp][col_key] = avg_mem

            print(
                f"  [{disp:<14}] {col_key:>8} | AEB: {100 * aeb:6.2f}% | "
                f"Time: {metrics['avg_time_sec'] * 1000:6.1f} ms | "
                f"Mem: {metrics['avg_peak_mem_kb']:6.1f} KiB"
            )


def render_generic_table(data_dict, fmt="{:.2f}", is_percentage=False):
    """Generates a table of a metric"""
    name_w = (
        max(len("Method"), *(len(d) for d in loaded))
        if loaded
        else len("Method")
    )
    col_w = max(10, max(len(c) for c in COLUMNS))
    head = f'{"Method":<{name_w}} | ' + " | ".join(
        f"{c:>{col_w}}" for c in COLUMNS
    )
    sep = "-" * len(head)
    rows = [head, sep]

    for disp in loaded:
        cells = []
        for c in COLUMNS:
            v = data_dict[disp].get(c)
            if v is None:
                cells.append(f'{"-":>{col_w}}')
            elif is_percentage:
                cells.append(f"{100 * v:>{col_w - 1}.2f}%")
            else:
                formatted_val = fmt.format(v)
                cells.append(f"{formatted_val:>{col_w}}")
        rows.append(f"{disp:<{name_w}} | " + " | ".join(cells))
    return "\n".join(rows)


report = [
    "=" * 78,
    "1. EXCESS OVER L1 LOWER BOUND (AEB %, Lower is Better)",
    "=" * 78,
    render_generic_table(results_aeb, is_percentage=True),
    "\n" + "=" * 78,
    "2. EXECUTION TIME PER INSTANCE (Milliseconds, Lower is Better)",
    "=" * 78,
    render_generic_table(
        {
            k: {col: v * 1000.0 for col, v in row.items()}
            for k, row in results_time.items()
        },
        fmt="{:.2f} ms",
    ),
    "\n" + "=" * 78,
    "3. PEAK MEMORY CONSUMPTION (KiB, Lower is Better)",
    "=" * 78,
    render_generic_table(results_mem, fmt="{:.1f} KiB"),
    "\n",
]

report_output = "\n".join(report)
print("\n" + report_output)

results_dir = os.path.join(HERE, "../results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, "benchmark_metrics.txt")

with open(results_path, "w", encoding="utf-8") as out:
    out.write(report_output)
print(f"Results successfully saved to {results_path}")