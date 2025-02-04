import math
import matplotlib.pyplot as plt

class AirHockeyAI:
    def __init__(self, table_width=500, table_height=1500, goal_width=200, mallet_speed=500, mallet_radius=30, puck_radius=15):
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

        # Goal positions
        self.goal_center = table_width / 2
        self.defensive_box = (
            (self.goal_center - goal_width / 2, 0),
            (self.goal_center + goal_width / 2, 100)
        )

    def calculate_velocity(self, position1, position2, time_step):
        """
        Calculate velocity based on two puck positions and the time step.
        :param position1: The initial position of the puck (x1, y1).
        :param position2: The final position of the puck (x2, y2).
        :param time_step: The time difference between the two positions (in seconds).
        :return: A tuple representing the velocity (vx, vy).
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
            # Update position
            px += vx * dt
            py += vy * dt


            # Check for wall collisions
            if px <= self.puck_radius or px >= self.table_width - self.puck_radius:
                vx = -vx  # Bounce off vertical walls
            if py <= self.puck_radius or py >= self.table_height - self.puck_radius:
                vy = -vy  # Bounce off horizontal walls

            # Append current position to trajectory
            trajectory.append((px, py))
            time_elapsed += dt

        return trajectory

    def calculate_collision(self, puck_position, puck_velocity, mallet_position, mallet_velocity):
        """
        Simulate collision between the mallet and the puck (two circles).
        """
        dx = puck_position[0] - mallet_position[0]
        dy = puck_position[1] - mallet_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance > self.puck_radius + self.mallet_radius:
            return puck_velocity  # No collision

        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance

        # Relative velocity
        rvx = mallet_velocity[0] - puck_velocity[0]
        rvy = mallet_velocity[1] - puck_velocity[1]
        relative_velocity = rvx * nx + rvy * ny

        if relative_velocity > 0:
            return puck_velocity  # No meaningful collision

        restitution = 0.9  # Bounciness factor
        impulse = -(1 + restitution) * relative_velocity
        new_vx = puck_velocity[0] + impulse * nx
        new_vy = puck_velocity[1] + impulse * ny

        return new_vx, new_vy

    def decide_action(self, puck_trajectory):
        """
        Decide the best action based on the predicted puck trajectory.
        """
        for px, py in puck_trajectory:
            if self.defensive_box[0][0] <= px <= self.defensive_box[1][0] and py <= self.defensive_box[1][1]:
                return "DEFEND"
        return "ATTACK"

    def visualize(self, puck_trajectory, mallet_position):
        """
        Visualize the puck trajectory, mallet position, and table setup.
        """
        plt.figure(figsize=(6, 12))
        plt.plot(*zip(*puck_trajectory), '-b', label="Puck Trajectory")
        plt.scatter(puck_trajectory[0][0], puck_trajectory[0][1], color='red', label="Starting Point")
        plt.scatter(mallet_position[0], mallet_position[1], color='orange', label="Mallet Position")
        plt.gca().add_patch(
            plt.Rectangle(
                self.defensive_box[0],
                self.goal_width,
                self.defensive_box[1][1] - self.defensive_box[0][1],
                edgecolor="green",
                fill=False,
                label="Defensive Box"
            )
        )
        plt.xlim(0, self.table_width)
        plt.ylim(0, self.table_height)
        plt.legend()
        plt.title("Air Hockey Simulation")
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.show()

# Example usage:
ai = AirHockeyAI()
pos1 = (350,1400) #need to get this form the simulation
pos2 = (330,1380)

time_step = 0.02

vx, vy = ai.calculate_velocity(pos1, pos2, time_step)
trajectory = ai.predict_puck_trajectory(pos2, (vx, vy))
ai.visualize(trajectory, ai.mallet_position)
