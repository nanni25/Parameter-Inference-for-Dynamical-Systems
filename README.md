# Automated Parameter Estimation in Biological Networks via Evolution Strategies

## Overview
This project presents an automated parameter estimation pipeline to discover unobservable kinetic constants in complex biological networks. Starting from static models, the algorithm programmatically transforms them into executable dynamical systems by injecting non-linear Ordinary Differential Equations and boundary constraints, enabling time-course simulation. 

Due to the stiffness and non-differentiability of the resulting systems, traditional gradient-based methods are ineffective. To solve this, I designed a derivative-free optimization algorithm based on Evolution Strategies, enhanced with Adam momentum tracking and exponential decay schedules, to efficiently recover parameters that reproduce target dynamics. 

The pipeline autonomously queries a Large Language Model (LLAMA-3.3-70B) to extract biological target concentrations, establishing the ground truth for the objective function. To evaluate how closely the optimized parameters match these LLM-generated targets, the underlying simulator leverages Just-In-Time LLVM compilation to efficiently evaluate candidate solutions. 

Experimental results show that the approach successfully recovers parameter sets that reproduce the target behavior, highlighting how AI methods can enable fully automated calibration of complex dynamical systems.

## Results
The `Results/` directory contains visual validations of the optimization pipeline, including:
* **Loss Convergence Graphs:** Demonstrating the performance and decay schedules of the Evolution Strategies algorithm over generations.
* **Target Comparisons:** Plots comparing the simulated steady-state means against the LLM-generated ground truth targets.

## Repository Structure
```text
├── Configurations/
│   └── config.json         # Hyperparameters for the optimizer
│   └── requirements.txt    # Dependencies
├── Cured Models/           # Dynamic models created by the algorithm
├── Models/                 # Static models
├── Results/                # Convergence graphs and target comparison plots
├── Targets/                # LLM query results
├── main.py                 # Main entry point and algorithm orchestrator
├── modifier.py             # Parses static SBML and injects ODEs/rules
├── optimizer.py            # Evolution Strategies and Adam momentum logic
├── simulation.py           # JIT LLVM compilation and loss evaluation
├── targets.py              # Autonomously queries LLAMA-3.3-70B via Groq
├── requirements.txt        # Project dependencies
└── README.md