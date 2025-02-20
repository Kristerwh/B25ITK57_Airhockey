import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

TABLE_WIDTH = 500
TABLE_HEIGHT = 1500
CYCLE_TIME = 0.01 #seconds
PREDICTION_MAX_STEPS = 100
MALLET_SPEED = 400
GOAL_WIDTH = TABLE_WIDTH * 0.4
DEFENSE_DEPTH = TABLE_HEIGHT * 0.1
MALLET_RADIUS = 30
PUCK_RADIUS = 20

class AirHockeyAI:
    def __init__(self, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT, mallet_speed = MALLET_SPEED, goal_width = GOAL_WIDTH, defense_depth = DEFENSE_DEPTH, mallet_radius = MALLET_RADIUS, puck_radius = PUCK_RADIUS ):
        self.table_width = table_width
        self.table_height = table_height
        self.last_puck_position = None
        self.current_puck_position = None
        self.last_mallet_position = None
        self.current_mallet_position = None
        self.goal_width = goal_width
        self.goal_center = table_width / 2
        self.defense_depth = defense_depth
        self.defensive_box = (
            (self.goal_center - self.goal_width / 2, 0),  # Bottom-left corner
            (self.goal_center + self.goal_width / 2, self.defense_depth),  # Top-right corner
        )
        self.mallet_position = (table_width / 2, self.defense_depth / 2)
        self.mallet_speed = mallet_speed
        self.puck_vx = None
        self.puck_vy = None
        self.mallet_vx = None
        self.mallet_vy = None
        self.mallet_radius = mallet_radius
        self.puck_radius = puck_radius
        self.puck_mass = 0.2  # grams
        self.mallet_mass = 0.5 # grams

    def update_puck_position(self, new_position):
        """
        Update the puck's position and calculate the velocity vector.
        :param new_position: Tuple (x, y) representing the puck's new position.
        :return: Tuple (vx, vy) representing the puck's velocity vector.
        """

        self.last_puck_position = self.current_puck_position
        self.current_puck_position = new_position

        if self.last_puck_position is not None:
            dx = self.current_puck_position[0] - self.last_puck_position[0]
            dy = self.current_puck_position[1] - self.last_puck_position[1]
            #Assuming 100hz system (1/100 = 0.01 seconds per cycle)
            vx = dx / CYCLE_TIME
            vy = dy / CYCLE_TIME
            self.puck_vx = vx
            self.puck_vy = vy
            return vx, vy
        return None, None #if we have no last puck position
    
    def update_mallet_position(self, new_position):
        """
        Update the mallet's position and calculate the velocity vector.
        :param new_position: Tuple (x, y) representing the mallet's new position.
        :return: Tuple (vx, vy) representing the mallet's velocity vector.
        """

        self.last_mallet_position = self.current_mallet_position
        self.current_mallet_position = new_position

        if self.last_mallet_position is not None:
            dx = self.current_mallet_position[0] - self.last_mallet_position[0]
            dy = self.current_mallet_position[1] - self.last_mallet_position[1]
            #Assuming 100hz system (1/100 = 0.01 seconds per cycle)
            vx = dx / CYCLE_TIME
            vy = dy / CYCLE_TIME
            self.mallet_vx = vx
            self.mallet_vy = vy
            return vx, vy
        return None, None #if we have no last mallet position

    def predict_trajectory(self,vx,vy,puck_x,puck_y, max_steps = PREDICTION_MAX_STEPS):
        """
        Predict the puck's trajectory until it intersects with the table boundaries or a specific line.
        :param vx: Puck's x velocity.
        :param vy: Puck's y velocity.
        :param puck_x: Current x position of the puck.
        :param puck_y: Current y position of the puck.
        :return: List of points (x, y) representing the puck's path.
        """

        trajectory = []
        current_x, current_y = puck_x, puck_y

        for _ in range(max_steps):
            # Add current position to trajectory
            trajectory.append((current_x, current_y))

            # Update the puck's position based on its velocity
            current_x += vx * CYCLE_TIME
            current_y += vy * CYCLE_TIME

            # Check for collisions with walls and adjust velocity accordingly
            if current_x <= self.puck_radius or current_x >= self.table_width - self.puck_radius:
                vx = -vx  # Reverse x velocity
            if current_y <= self.puck_radius * 2 or current_y >= self.table_height - (self.puck_radius * 2):
                vy = -vy  # Reverse y velocity
        return trajectory

    def handle_collision(self, puck_x, puck_y, mallet_x, mallet_y):
        dx = mallet_x - puck_x
        dy = mallet_y - puck_y
        dist = math.sqrt(dx**2 + dy**2)

        # **Step 1: Check if collision occurs (distance is less than sum of radii)**
        if dist < (self.puck_radius + self.mallet_radius):
            # **Step 2: Compute collision angle**
            angle = math.atan2(dy, dx)
            sin_a, cos_a = math.sin(angle), math.cos(angle)

            # **Step 3: Rotate velocities into collision frame**
            puck_vx_rot = self.puck_vx * cos_a + self.puck_vy * sin_a
            puck_vy_rot = self.puck_vy * cos_a - self.puck_vx * sin_a
            mallet_vx_rot = self.mallet_vx * cos_a + self.mallet_vy * sin_a
            # mallet_vy_rot = self.mallet_vy * cos_a - self.mallet_vx * sin_a

            # **Step 4: Compute new velocities after collision using 1D elastic collision equations**
            puck_vx_rot_final = ((self.puck_mass - self.mallet_mass) * puck_vx_rot + 2 * self.mallet_mass * mallet_vx_rot) / (self.puck_mass + self.mallet_mass)
            # mallet_vx_rot_final = ((self.mallet_mass - self.puck_mass) * mallet_vx_rot + 2 * self.puck_mass * puck_vx_rot) / (self.puck_mass + self.mallet_mass)

            # **Step 5: Rotate velocities back to 2D space**
            self.puck_vx = puck_vx_rot_final * cos_a - puck_vy_rot * sin_a
            self.puck_vy = puck_vy_rot * cos_a + puck_vx_rot_final * sin_a
            # self.mallet_vx = mallet_vx_rot_final * cos_a - mallet_vy_rot * sin_a
            # self.mallet_vy = mallet_vy_rot * cos_a + mallet_vx_rot_final * sin_a
            print(f"🎯 Collision occurred! New Puck Velocity: ({self.puck_vx:.2f}, {self.puck_vy:.2f})")

    def predict_puck_after_contact(self, intercept_point, vx, vy):
        """
        Predict the puck's trajectory after the mallet makes contact at the intercept point.
        """
        # Assuming a simple reflection of the puck's trajectory
        ix, iy = intercept_point
        #vx, vy = self.update_puck_position(intercept_point)  # Current puck velocity
        return self.predict_trajectory(vx, -vy, ix, iy)

    def calculate_intercept_points(self, puck_trajectory):
        """
        Calculate intercept points where the mallet can reach the puck in time,
        considering the radii of both the puck and mallet.
        """
        intercept_points = []

        for step, (px, py) in enumerate(puck_trajectory):
            # Distance from the mallet to the intercept point
            dx = px - self.mallet_position[0]
            dy = py - self.mallet_position[1]
            dist = math.sqrt(dx**2 + dy**2)

            # Calculate time required for the mallet to reach this point
            time_to_reach = dist / self.mallet_speed

            # Check if the mallet can reach in time AND if a valid collision would occur
            if time_to_reach <= step * CYCLE_TIME and dist > (self.puck_radius + self.mallet_radius):
                # **Move the mallet towards the intercept point but stop at valid collision distance**
                new_mallet_x = self.mallet_position[0] + (dx / dist) * (dist - (self.puck_radius + self.mallet_radius))
                new_mallet_y = self.mallet_position[1] + (dy / dist) * (dist - (self.puck_radius + self.mallet_radius))

                # Update the mallet's position **towards** the intercept point without overlapping
                self.update_mallet_position((new_mallet_x, new_mallet_y))

                # **Simulate a collision when the mallet reaches the adjusted position**
                self.handle_collision(px, py, new_mallet_x, new_mallet_y)

                # **After the collision, update the puck's trajectory with new velocity**
                new_trajectory = self.predict_puck_after_contact((px, py), self.puck_vx, self.puck_vy)

                return intercept_points, new_trajectory  # Return first valid intercept

        return intercept_points, puck_trajectory  # If no collision happens, return original trajectory



    def visualize_ai_and_trajectory(self,trajectory, col_trajectory, defensive_box, mallet_position, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT,):
        """
        Visualize the puck's predicted trajectory along with the AI's defensive box and mallet position.
        :param trajectory: List of points (x, y) representing the puck's trajectory.
        :param table_width: Width of the air hockey table.
        :param table_height: Height of the air hockey table.
        :param defensive_box: Tuple ((x_min, y_min), (x_max, y_max)) defining the defensive box.
        :param mallet_position: Tuple (x, y) representing the mallet's position.
        """

        # Unpack the trajectory into x and y coordinates
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        x1_coords = [point[0] for point in col_trajectory]
        y1_coords = [point[1] for point in col_trajectory]

        # Plot the table boundary
        plt.figure(figsize=(4, 8))
        plt.plot(x_coords, y_coords, label="Puck Trajectory", color="blue", marker="o", markersize=3)
        plt.plot(x1_coords, y1_coords, label="Puck Trajectory1", color="red", marker="o", markersize=3)

        if trajectory:
            plt.scatter(trajectory[0][0], trajectory[0][1], color="red", label="Starting Point", zorder=5, s=50)

        box_x_min, box_y_min = defensive_box[0]
        box_x_max, box_y_max = defensive_box[1]
        plt.gca().add_patch(
            plt.Rectangle((box_x_min, box_y_min), box_x_max - box_x_min, box_y_max - box_y_min,
                          edgecolor="green", facecolor="none", linewidth=2, label="Defensive Box")
        )

        plt.scatter(mallet_position[0], mallet_position[1], color="orange", label="Mallet Position", zorder=5, s=100)

        # Draw the table boundaries
        plt.axhline(0, color="black", linestyle="--")  # Bottom boundary
        plt.axhline(table_height, color="black", linestyle="--")  # Top boundary
        plt.axvline(0, color="black", linestyle="--")  # Left boundary
        plt.axvline(table_width, color="black", linestyle="--")  # Right boundary

        # Label and style
        plt.title("Air Hockey Puck Trajectory")
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.xlim(0, table_width)
        plt.ylim(0, table_height)
        plt.legend()
        plt.grid()
        plt.show()

ai = AirHockeyAI()
## Simulated puck positions for trajectory prediction
positions = [
    (100, 625),  # Initial position
    (110, 600),  # Second position
]

# Step 1: Update puck position and calculate velocity
for pos in positions:
    vx, vy = ai.update_puck_position(pos)

trajectory = ai.predict_trajectory(vx, vy, positions[-1][0], positions[-1][1])

intercept, new_trajectory = ai.calculate_intercept_points(trajectory)

ai.visualize_ai_and_trajectory(trajectory, new_trajectory, ai.defensive_box, ai.mallet_position)