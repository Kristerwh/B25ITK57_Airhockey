import math
import matplotlib.pyplot as plt

TABLE_WIDTH = 500
TABLE_HEIGHT = 1500
CYCLE_TIME = 0.01 #seconds
PREDICTION_MAX_STEPS = 100
MALLET_SPEED = 400
GOAL_WIDTH = TABLE_WIDTH * 0.4
DEFENSE_DEPTH = TABLE_HEIGHT * 0.1


class AirhockeyAI:
    def __init__(self, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT, mallet_speed = MALLET_SPEED, goal_width = GOAL_WIDTH, defense_depth = DEFENSE_DEPTH ):
        self.table_width = table_width
        self.table_height = table_height
        self.last_puck_position = None
        self.current_puck_position = None
        self.goal_width = goal_width
        self.goal_center = table_width / 2
        self.defense_depth = defense_depth
        self.defensive_box = (
            (self.goal_center - self.goal_width / 2, 0),  # Bottom-left corner
            (self.goal_center + self.goal_width / 2, self.defense_depth),  # Top-right corner
        )
        self.mallet_position = (table_width / 2, self.defense_depth / 2)
        self.mallet_speed = mallet_speed
        self.vx = None
        self.vy = None

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
            self.vx = vx
            self.vy = vy
            return vx, vy
        return None, None #if we have no last puck position

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
            if current_x <= 0 or current_x >= self.table_width:
                vx = -vx  # Reverse x velocity
            if current_y <= 0 or current_y >= self.table_height:
                vy = -vy  # Reverse y velocity

        return trajectory

    def calculate_intercept_points(self, puck_trajectory):
        """
        Calculate three possible intercept points for the mallet based on the puck's trajectory.
        """
        intercept_points = []
        for step, (px, py) in enumerate(puck_trajectory):
            # Calculate time to reach the intercept point
            time_to_reach = abs(py - self.mallet_position[1]) / self.mallet_speed

            # If the mallet can reach the point in time, consider it
            if time_to_reach <= step * 0.01:
                intercept_points.append((px, py))
                if len(intercept_points) == 3:
                    break

        return intercept_points

    def predict_puck_after_contact(self, intercept_point, vx, vy):
        """
        Predict the puck's trajectory after the mallet makes contact at the intercept point.
        """
        # Assuming a simple reflection of the puck's trajectory
        ix, iy = intercept_point
        #vx, vy = self.update_puck_position(intercept_point)  # Current puck velocity
        return self.predict_trajectory(vx, -vy, ix, iy)  # Reverse Y velocity to simulate rebound

    def decide_best_move(self, intercept_points):
        """
        Decide the best move by predicting puck trajectories after contact and evaluating each.
        """
        best_move = None
        best_score_distance = float("inf")

        # Define opponent's defensive box
        opponent_defensive_box = (
            (self.goal_center - self.goal_width / 2, self.table_height * 0.8),
            (self.goal_center + self.goal_width / 2, self.table_height)
        )

        for point in intercept_points:
            # Predict puck trajectory after contact at the intercept point
            trajectory = self.predict_puck_after_contact(point, self.vx, self.vy)

            # Evaluate the trajectory's proximity to the opponent's goal
            goal_center_x = (opponent_defensive_box[0][0] + opponent_defensive_box[1][0]) / 2
            goal_center_y = (opponent_defensive_box[0][1] + opponent_defensive_box[1][1]) / 2

            # Track how close the puck gets to the goal throughout the trajectory
            min_distance_to_goal = float("inf")
            for px, py in trajectory:
                # Check if the puck enters the opponent's defensive box
                if opponent_defensive_box[0][0] <= px <= opponent_defensive_box[1][0] and py >= \
                        opponent_defensive_box[0][1]:
                    # Prioritize trajectories that actually enter the goal box
                    return point

                # Otherwise, compute the distance to the goal center
                distance_to_goal = ((px - goal_center_x) ** 2 + (py - goal_center_y) ** 2) ** 0.5
                min_distance_to_goal = min(min_distance_to_goal, distance_to_goal)

            # Select the intercept point that results in the closest trajectory to the goal
            if min_distance_to_goal < best_score_distance:
                best_score_distance = min_distance_to_goal
                best_move = point

        return best_move

    """def decide_action(self, trajectory):
    
        Decide the AI's action based on the puck's trajectory.
        :param trajectory: List of points (x, y) representing the puck's predicted path.
        :return: String representing the action: "defend", "attack", or "do nothing".
        
        if not trajectory:
            return "do nothing"

        box_x_min, box_y_min = self.defensive_box[0]
        box_x_max, box_y_max = self.defensive_box[1]

        # Check if the puck is heading toward the defensive box
        for point in trajectory:
            if box_x_min <= point[0] <= box_x_max and box_y_min <= point[1] <= box_y_max:
                return "defend"

        # Check if there's an opportunity to attack
        if trajectory[-1][1] > self.defense_depth and trajectory[-1][1] > self.table_height * 0.8:
            return "attack"

        # Default to doing nothing
        return "do nothing"
    """
    def visualize_ai_and_trajectory(self,trajectory, defensive_box, mallet_position, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT,):
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

        # Plot the table boundary
        plt.figure(figsize=(4, 8))
        plt.plot(x_coords, y_coords, label="Puck Trajectory", color="blue", marker="o", markersize=3)

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










#---------------------------------------------------------------------


ai = AirhockeyAI()

"""
while true
positions = []
pos = get_pos #need to get the position of the puck form the simulation environment 
positions.append(pos)
vx,vy = ai.update_puck_position(pos)
if vx && vy is NUll
    pos = get_pos #need to get the position of the puck form the simulation environment 
    positions.append(pos)
    vx,vy = ai.update_puck_position(pos)
    
    
trajectory = ai.predict_trajectory(vx, vy, positions[-1][0], positions[-1][1])
intercept_points = ai.calculate_intercept_points(trajectory)

for i, intercept_point in enumerate(intercept_points):
    post_contact_trajectory = ai.predict_puck_after_contact(intercept_point,vx,vy)
best_action = ai.decide_best_move(intercept_points)
print(f"Best Action: {best_action}")
"""

## Simulated puck positions for trajectory prediction
positions = [
    (100, 625),  # Initial position
    (110, 600),  # Second position
]

# Step 1: Update puck position and calculate velocity
for pos in positions:
    vx, vy = ai.update_puck_position(pos)

# Step 2: Predict puck trajectory
trajectory = ai.predict_trajectory(vx, vy, positions[-1][0], positions[-1][1])

ai.visualize_ai_and_trajectory(trajectory, ai.defensive_box, ai.mallet_position)

input("Press Enter to continue...")

# Step 3: Calculate intercept points
intercept_points = ai.calculate_intercept_points(trajectory)
print(f"Intercept Points: {intercept_points}")

# Step 4: Simulate puck trajectory after contact for each intercept point
for i, intercept_point in enumerate(intercept_points):
    post_contact_trajectory = ai.predict_puck_after_contact(intercept_point,vx,vy)

    print(post_contact_trajectory)

    # Visualize each trajectory
    print(f"Post-contact trajectory for intercept point {i + 1}: {intercept_point}")
    ai.visualize_ai_and_trajectory(post_contact_trajectory, ai.defensive_box, intercept_point)

    input("Press Enter to continue...")

# Step 5: Decide the best action (based on scoring potential)
best_action = ai.decide_best_move(intercept_points)
print(f"Best Action: {best_action}")