import math
import matplotlib.pyplot as plt

class AirHockeyAI:
    def __init__(self, table_width=500, table_height=1500, goal_width=200, 
                 mallet_speed=0.1, mallet_radius=30, puck_radius=15, stray_distance=1):
        """
        Initialize the AI with table dimensions, mallet and puck properties.
        """
        self.table_width = table_width  # mm
        self.table_height = table_height  # mm
        self.goal_width = goal_width  # mm

        # Mallet properties
        self.mallet_speed = mallet_speed  # mm/s
        self.mallet_radius = mallet_radius  # mm
        self.mallet_position = (table_width / 2, mallet_radius)  # Start at center of defensive zone

        # Puck properties
        self.puck_radius = puck_radius

        # Stray Distance (Tiny deviation left, right, and down)
        self.stray_distance = stray_distance  # mm

    def calculate_velocity(self, position1, position2, time_step):
        """
        Calculate velocity based on two puck positions and the time step.
        """
        x1, y1 = position1
        x2, y2 = position2

        vx = (x2 - x1) / time_step
        vy = (y2 - y1) / time_step

        return vx, vy

    def predict_puck_trajectory(self, puck_position, puck_velocity, max_time=2):
        """
        Predict the puck's trajectory based on its position and velocity, considering table boundaries.
        """
        trajectory = []
        px, py = puck_position
        vx, vy = puck_velocity

        dt = 0.01  # Time step (100 Hz)
        time_elapsed = 0

        while time_elapsed < max_time:
            px += vx * dt
            py += vy * dt

            # Check for wall collisions
            if px <= self.puck_radius or px >= self.table_width - self.puck_radius:
                vx = -vx  # Bounce off vertical walls
            if py <= self.puck_radius or py >= self.table_height - self.puck_radius:
                vy = -vy  # Bounce off horizontal walls

            trajectory.append((px, py))
            time_elapsed += dt

        return trajectory

    def calculate_intercept_points(self, puck_trajectory):
        """
        Calculate three possible intercept points slightly down, left, and right.
        """
        intercept_points = []
        
        for step, (px, py) in enumerate(puck_trajectory):
            straydist = self.stray_distance
            # Calculate time to reach the intercept point
            time_to_reach = abs(py - self.mallet_position[1]) / self.mallet_speed

            # If the mallet can reach the point in time, consider it
            if time_to_reach <= step * 0.01:
                # Generate three intercepts (center, left, right)
                for i in range(0, straydist):
                    intercepts = [
                        (px, py),  # Center
                        (max(px - self.stray_distance, self.mallet_radius), py + straydist),  # Left-shifted
                        (min(px + self.stray_distance, self.table_width - self.mallet_radius), py + straydist)  # Right-shifted
                    ]
                    intercept_points.extend(intercepts)

        return intercept_points  # Return only three points

    def predict_puck_after_contact(self, intercept_point, vx, vy):
        """
        Predict the puck's trajectory after making contact at an intercept point.
        """
        ix, iy = intercept_point
        new_velocity = (-vx, -vy)  # Simulate a simple rebound (flipping Y velocity)

        return self.predict_puck_trajectory((ix, iy), new_velocity)  # Start at the correct position

    def visualize(self, puck_trajectory, mallet_position, intercept_points, predicted_trajectories):
        """
        Visualize the puck trajectory, mallet position, intercept positions, and post-impact trajectories.
        """
        plt.figure(figsize=(10, 12))

        
        # Plot original puck trajectory
        plt.plot(*zip(*puck_trajectory), '-b', label="Puck Trajectory Before Impact")
        plt.scatter(puck_trajectory[0][0], puck_trajectory[0][1], color='red', label="Starting Point")
        plt.scatter(mallet_position[0], mallet_position[1], color='orange', label="Mallet Position")

        # Plot intercept points
        colors = ["green", "purple", "brown"]
        labels = ["Intercept Center", "Intercept Left", "Intercept Right"]
        for i, intercept in enumerate(intercept_points):
            plt.scatter(intercept[0], intercept[1], label=labels[i%3])

        # Plot post-impact trajectories
        for i, trajectory in enumerate(predicted_trajectories):
            if trajectory:
                plt.plot(*zip(*trajectory), linestyle="dashed", label=f"After Impact {labels[i%3]}")

        plt.xlim(0, self.table_width)
        plt.ylim(0, self.table_height)
        plt.legend()
        plt.title("Air Hockey Simulation with Tiny Stray & Post-Impact Prediction")
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# Example usage:
ai = AirHockeyAI()
pos1 = (350, 1400)  # Initial puck position
pos2 = (330, 1380)  # Next observed puck position
time_step = 0.02

# Compute initial puck velocity
vx, vy = ai.calculate_velocity(pos1, pos2, time_step)

# Predict puck trajectory before contact
trajectory = ai.predict_puck_trajectory(pos2, (vx, vy))

# Find three alternative intercept points
intercepts = ai.calculate_intercept_points(trajectory)

# Predict post-impact trajectories for each intercept
predicted_paths = [ai.predict_puck_after_contact(ip, vx, vy) for ip in intercepts]

# Visualize everything
ai.visualize(trajectory, ai.mallet_position, intercepts, predicted_paths)


