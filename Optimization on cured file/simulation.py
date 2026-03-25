import roadrunner
import numpy as np
import config

# 1. INITIALIZE THE SIMULATOR
print(f"Loading model from: {config.MODEL_PATH}")
rr = roadrunner.RoadRunner(config.MODEL_PATH)

def evaluate_loss(theta):
    """
    Injects the given parameters into the model, runs the simulation,
    and calculates the normalized sum of squared errors (Loss).
    
    Args:
        theta (np.ndarray): The parameter vector guessed by the optimizer (in log-space).
        
    Returns:
        float: The calculated Loss J.
    """

    # 2. RESET THE MODEL
    rr.resetAll()
    
    # 3. PARAMETERS
    """Problem: we could have negative kinetic costants by drawing noise from the distribution! 
       we could solve it by forcing positive values with e^theta"""
    actual_params = np.exp(theta)
    
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, actual_params):
        rr.setValue(param_id, param_val)
        
    # 4. RUN THE SIMULATION
    rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS)
    
    # 5. EXTRACT RESULTS AND CALCULATE LOSS
    final_y = np.array([rr.getValue(var_id) for var_id in config.MEAN_VARIABLES])

    # We add an extremely small number (epsilon) to the denominator to prevent a 
    # "Divide by Zero" crash in case a biological target M is ever exactly 0.
    epsilon = 1e-8
    loss = np.sum(((final_y - config.TARGET_M) / (config.TARGET_M + epsilon))**2)
    
    return loss