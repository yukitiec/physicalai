import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def plot_observations(observations, save_path=None):
    """
    Plot all observation results using subplots
    
    Parameters:
    observations: dict containing the observation data
    save_path: optional path to save the plot
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5])
    
    # Get time steps
    num_steps = observations["pos_robot"].shape[0]
    time_steps = np.arange(num_steps)
    
    # 1. Position plots (Robot and Human on same axes)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Robot position
    pos_robot = observations["pos_robot"]
    ax1.plot(time_steps, pos_robot[:, 0], 'b-', linewidth=2, label='Robot X', alpha=0.8)
    ax1.plot(time_steps, pos_robot[:, 1], 'b--', linewidth=2, label='Robot Y', alpha=0.8)
    ax1.plot(time_steps, pos_robot[:, 2], 'b:', linewidth=2, label='Robot Z', alpha=0.8)
    
    # Human position
    pos_human = observations["pos_human"]
    ax1.plot(time_steps, pos_human[:, 0], 'r-', linewidth=2, label='Human X', alpha=0.8)
    ax1.plot(time_steps, pos_human[:, 1], 'r--', linewidth=2, label='Human Y', alpha=0.8)
    ax1.plot(time_steps, pos_human[:, 2], 'r:', linewidth=2, label='Human Z', alpha=0.8)
    
    ax1.set_title('Robot and Human Positions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Position (m)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity plots (Robot and Human on same axes)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Robot velocity
    vel_robot = observations["vel_robot"]
    ax2.plot(time_steps, vel_robot[:, 0], 'b-', linewidth=2, label='Robot VX', alpha=0.8)
    ax2.plot(time_steps, vel_robot[:, 1], 'b--', linewidth=2, label='Robot VY', alpha=0.8)
    ax2.plot(time_steps, vel_robot[:, 2], 'b:', linewidth=2, label='Robot VZ', alpha=0.8)
    
    # Human velocity
    vel_human = observations["vel_human"]
    ax2.plot(time_steps, vel_human[:, 0], 'r-', linewidth=2, label='Human VX', alpha=0.8)
    ax2.plot(time_steps, vel_human[:, 1], 'r--', linewidth=2, label='Human VY', alpha=0.8)
    ax2.plot(time_steps, vel_human[:, 2], 'r:', linewidth=2, label='Human VZ', alpha=0.8)
    
    ax2.set_title('Robot and Human Velocities', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rope middle point
    ax3 = fig.add_subplot(gs[2, 0])
    
    rope_mid = observations["rope_mid_point"]
    ax3.plot(time_steps, rope_mid[:, 0], 'g-', linewidth=2, label='Rope Mid X', alpha=0.8)
    ax3.plot(time_steps, rope_mid[:, 1], 'g--', linewidth=2, label='Rope Mid Y', alpha=0.8)
    ax3.plot(time_steps, rope_mid[:, 2], 'g:', linewidth=2, label='Rope Mid Z', alpha=0.8)
    
    ax3.set_title('Rope Middle Point Position', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Position (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Link states (first few links)
    ax4 = fig.add_subplot(gs[2, 1])
    
    link_state = observations["link_state"]
    num_links = min(5, link_state.shape[1])  # Plot first 5 links
    
    for i in range(num_links):
        # Plot position of each link
        ax4.plot(time_steps, link_state[:, i, 0], linewidth=1.5, 
                label=f'Link {i+1} X', alpha=0.7)
        ax4.plot(time_steps, link_state[:, i, 1], linewidth=1.5, 
                label=f'Link {i+1} Y', alpha=0.7, linestyle='--')
        ax4.plot(time_steps, link_state[:, i, 2], linewidth=1.5, 
                label=f'Link {i+1} Z', alpha=0.7, linestyle=':')
    
    ax4.set_title('Rope Link Positions (First 5 Links)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Position (m)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_3d_trajectories(observations, save_path=None):
    """
    Plot 3D trajectories of robot, human, and rope middle point
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Robot trajectory
    pos_robot = observations["pos_robot"]
    ax.plot(pos_robot[:, 0], pos_robot[:, 1], pos_robot[:, 2], 
            'b-', linewidth=3, label='Robot Trajectory', alpha=0.8)
    
    # Human trajectory
    pos_human = observations["pos_human"]
    ax.plot(pos_human[:, 0], pos_human[:, 1], pos_human[:, 2], 
            'r-', linewidth=3, label='Human Trajectory', alpha=0.8)
    
    # Rope middle point trajectory
    rope_mid = observations["rope_mid_point"]
    ax.plot(rope_mid[:, 0], rope_mid[:, 1], rope_mid[:, 2], 
            'g-', linewidth=2, label='Rope Middle Point', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(pos_robot[0, 0], pos_robot[0, 1], pos_robot[0, 2], 
              c='blue', s=100, marker='o', label='Robot Start')
    ax.scatter(pos_robot[-1, 0], pos_robot[-1, 1], pos_robot[-1, 2], 
              c='blue', s=100, marker='s', label='Robot End')
    
    ax.scatter(pos_human[0, 0], pos_human[0, 1], pos_human[0, 2], 
              c='red', s=100, marker='o', label='Human Start')
    ax.scatter(pos_human[-1, 0], pos_human[-1, 1], pos_human[-1, 2], 
              c='red', s=100, marker='s', label='Human End')
    
    ax.set_title('3D Trajectories: Robot, Human, and Rope', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    plt.show()

def plot_distance_analysis(observations, save_path=None):
    """
    Plot distance analysis between robot, human, and rope
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distance Analysis', fontsize=16, fontweight='bold')
    
    num_steps = observations["pos_robot"].shape[0]
    time_steps = np.arange(num_steps)
    
    # Calculate distances
    pos_robot = observations["pos_robot"]
    pos_human = observations["pos_human"]
    rope_mid = observations["rope_mid_point"]
    
    # Robot-Human distance
    robot_human_dist = np.linalg.norm(pos_robot - pos_human, axis=1)
    
    # Robot-Rope distance
    robot_rope_dist = np.linalg.norm(pos_robot - rope_mid[:, :3], axis=1)
    
    # Human-Rope distance
    human_rope_dist = np.linalg.norm(pos_human - rope_mid[:, :3], axis=1)
    
    # Plot distances
    axes[0, 0].plot(time_steps, robot_human_dist, 'purple', linewidth=2)
    axes[0, 0].set_title('Robot-Human Distance')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_steps, robot_rope_dist, 'blue', linewidth=2)
    axes[0, 1].set_title('Robot-Rope Distance')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Distance (m)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time_steps, human_rope_dist, 'red', linewidth=2)
    axes[1, 0].set_title('Human-Rope Distance')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # All distances on same plot
    axes[1, 1].plot(time_steps, robot_human_dist, 'purple', linewidth=2, label='Robot-Human')
    axes[1, 1].plot(time_steps, robot_rope_dist, 'blue', linewidth=2, label='Robot-Rope')
    axes[1, 1].plot(time_steps, human_rope_dist, 'red', linewidth=2, label='Human-Rope')
    axes[1, 1].set_title('All Distances')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Distance (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance analysis saved to {save_path}")
    
    plt.show()

def plot_all_observations(saveDir, observations):
    """
    Plot all observation data and save as .png files in the specified directory
    
    Parameters:
    saveDir: directory path to save the plot images
    observations: dict containing the observation data
    """
    print(f"Plotting observation data and saving to {saveDir}...")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        print(f"Created directory: {saveDir}")
    
    # Main observation plots
    main_plot_path = os.path.join(saveDir, "observations_main.png")
    plot_observations(observations, main_plot_path)
    
    # 3D trajectory plot
    trajectory_plot_path = os.path.join(saveDir, "trajectories_3d.png")
    plot_3d_trajectories(observations, trajectory_plot_path)
    
    # Distance analysis
    distance_plot_path = os.path.join(saveDir, "distance_analysis.png")
    plot_distance_analysis(observations, distance_plot_path)
    
    print(f"All plots saved to {saveDir}:")
    print(f"  - {os.path.basename(main_plot_path)}")
    print(f"  - {os.path.basename(trajectory_plot_path)}")
    print(f"  - {os.path.basename(distance_plot_path)}")
    print("Plotting completed!")

# Example usage
if __name__ == "__main__":
    # Example data structure (replace with your actual data)
    example_observations = {
        "link_state": np.random.randn(100, 10, 6),  # (time_steps, num_links, 6)
        "rope_mid_point": np.random.randn(100, 7),  # (time_steps, 7)
        "pos_robot": np.random.randn(100, 3),       # (time_steps, 3)
        "vel_robot": np.random.randn(100, 3),       # (time_steps, 3)
        "pos_human": np.random.randn(100, 3),       # (time_steps, 3)
        "vel_human": np.random.randn(100, 3)        # (time_steps, 3)
    }
    
    # Example usage
    save_directory = "./observation_plots"
    plot_all_observations(save_directory, example_observations) 