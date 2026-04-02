import numpy as np
import config
import matplotlib.pyplot as plt
from simulation import evaluate_loss, generate_synthetic_targets

def plot_results(history_best_loss, history_mean_loss, num_params, final_params):
    """
    Generates and saves thesis-ready visualizations of the optimization run.
    """
    # 1. Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_mean_loss, label='Mean Loss (Population)', color='#1f77b4', alpha=0.8)
    plt.plot(history_best_loss, label='Best Loss (Individual)', color='#d62728', linewidth=2)
    
    plt.yscale('log')
    plt.title('Optimizer Convergence over Generations', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('convergence_plot.pdf', dpi=300)
    
    # 2. Parameter Comparison Plot
    plt.figure(figsize=(14, 7))
    x = np.arange(num_params) 
    width = 0.35               
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, config.TRUE_PARAMS, width, label='True Values', color='#2ca02c')
    rects2 = ax.bar(x + width/2, final_params, width, label='Optimized Values', color='#ff7f0e')

    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Comparison of True vs. Optimized Kinetic Parameters', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config.PARAMS_TO_OPTIMIZE, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('parameter_comparison.pdf', dpi=300)
    plt.show()
    
def main():
    np.random.seed(11)

    print("Generating pure synthetic targets (no noise)...")
    targets = generate_synthetic_targets(config.TRUE_PARAMS)
    
    num_params = len(config.PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    
    pop_size = config.POPULATION_SIZE
    half_pop = pop_size // 2 
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    
    initial_lr = config.LEARNING_RATE
    initial_sigma = config.SIGMA
    
    # Lists to track data for the convergence plot
    history_best_loss = []
    history_mean_loss = []
    
    print(f"Starting for {config.NUM_GENERATIONS} gens. Pop Size: {pop_size}")
    
    for epoch in range(config.NUM_GENERATIONS):
        decay_factor = np.exp(-3.0 * (epoch / config.NUM_GENERATIONS)) 
        current_lr = initial_lr * decay_factor
        current_sigma = max(initial_sigma * decay_factor, 0.001) 
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(pop_size)
        for i in range(pop_size):
            theta_try = theta + current_sigma * epsilons[i]
            raw_losses[i] = evaluate_loss(theta_try, targets)
            
        losses = np.zeros(pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, pop_size)
        
        step = np.dot(losses, epsilons) / (pop_size * current_sigma)
        g = -step
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + adam_epsilon)
        
        min_loss = np.min(raw_losses)
        mean_loss = np.mean(raw_losses)
        
        # Save the epoch's data for plotting
        history_best_loss.append(min_loss)
        history_mean_loss.append(mean_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Gen {epoch+1:03d}/{config.NUM_GENERATIONS} | LR: {current_lr:.4f} | Sig: {current_sigma:.4f} | Best Loss: {min_loss:.6f}")
        
    print("\nOptimization Complete:")
    final_params = np.exp(theta)
    
    print(f"{'Parameter':<15} | {'True Value':<12} | {'Optimized Value'}")
    print("-" * 50)
    for name, true_val, opt_val in zip(config.PARAMS_TO_OPTIMIZE, config.TRUE_PARAMS, final_params):
        print(f"{name:<15} | {true_val:<12.6f} | {opt_val:.6f}")

    # Call our dedicated plotting function
    plot_results(history_best_loss, history_mean_loss, num_params, final_params)

if __name__ == "__main__":
    main()