# Quality-Diversity Memetic Algorithm for NP-Hard Problems

[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-amber.svg)](https://github.com/ssergioogj/qd-heuristic-generation)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automating heuristic discovery for NP-hard combinatorial optimization problems using a Quality-Diversity (QD) evolutionary algorithm.

> **Note:** This repository corresponds to my Undergraduate Bachelor's Thesis (TFG) in Computer Science and Engineering at Universidad Carlos III de Madrid (UC3M) and is currently **in progress**.

## Overview

Discovering effective heuristics for complex combinatorial search spaces is traditionally a manual, domain-specific task. This project explores the use of illumination algorithms (**MAP-Elites**) augmented with a memetic local search based on ReEvo to:

- Maintain a diverse archive of high-performing heuristic policies across behavioral features.
- Overcome deceptive fitness landscapes in NP-hard combinatorial spaces.
- Benchmark automatically generated heuristics against standard algorithms and baselines.

## Project Structure

qd-heuristic-generation/
├── benchmark/
│   ├── online_bin_packing_1d/     # 1D Online Bin Packing benchmark (EoH-based)
│   └── utils/
│       └── visualizer.py          # Metric visualization
├── map_elites_reevo/
│   ├── llm/                       # LLM client interface for prompt-driven search
│   ├── reevo/                     # ReEvo core evolutionary loop
│   ├── evaluator.py               # Performance and runtime evaluation pipeline
│   ├── gnode.py                   # Graph-based node representation for MAP-Elites
│   ├── map_elites.py              # Core MAP-Elites 
│   ├── mutator.py                 # Local search and mutation operators
│   └── test_mutator.py            # Temp tests for mutation operators
├── .gitignore
└── README.md
