import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AirHockeyAI:
    def __init__(self, table_width=500, table_height=1500, goal_width=200, 
                 mallet_radius=30, puck_radius=15, mallet_speed=500):
        # Table
        self.table_width = table_width  # mm
        self.table_height = table_height  # mm
        self.goal_width = goal_width  # mm

        # Mallet
        self.mallet_radius = mallet_radius  # mm
        self.mallet_position = (table_width / 2, mallet_radius)  # Start at center of defensive zone
        self.mallet_speed = mallet_speed  

        # Puck
        self.puck_radius = puck_radius

    

    def predict_puck_trajectory(self, puck_position, angle, max_time=2, speed=500):
        trajectory = []

        # Starting point
        px, py = puck_position

        # Convert angle to radians and compute velocity
        angle_rad = math.radians(angle)
        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad)

        # Time step
        dt = 0.01
        time_elapsed = 0

        while time_elapsed < max_time:
            px += vx * dt
            py += vy * dt

            # Ensure puck bounces when its edge reaches the wall
            if px - self.puck_radius <= 0 or px + self.puck_radius >= self.table_width:
                # Bounce off left or right walls
                vx = -vx  
                # Keep inside bounds
                px = max(self.puck_radius, min(self.table_width - self.puck_radius, px))
            if py - self.puck_radius <= 0 or py + self.puck_radius >= self.table_height:
                # Bounce off top or bottom walls
                vy = -vy
                py = max(self.puck_radius, min(self.table_height - self.puck_radius, py))

            trajectory.append((px, py))
            time_elapsed += dt

        return trajectory

    
    def compute_contact_point(self, puck_pos, mallet_pos):
        # Compute direction vector from mallet center to puck center
        direction_x = puck_pos[0] - mallet_pos[0]
        direction_y = puck_pos[1] - mallet_pos[1]

        # Normalize direction vector
        magnitude = math.sqrt(direction_x**2 + direction_y**2) + 1e-6  # Avoid zero division
        direction_x /= magnitude
        direction_y /= magnitude

        # Calculate actual contact point by moving from the mallet center outward
        contact_x = mallet_pos[0] + direction_x * self.mallet_radius
        contact_y = mallet_pos[1] + direction_y * self.mallet_radius

        return (contact_x, contact_y)


    def compute_bounce_angle(puck_pos, mallet_pos, puck_velocity):
        # Compute normal vector (direction from mallet to puck)
        normal_x = puck_pos[0] - mallet_pos[0]
        normal_y = puck_pos[1] - mallet_pos[1]
        
        # Normalize the normal vector
        normal_mag = math.sqrt(normal_x ** 2 + normal_y ** 2) + 1e-6  # Avoid zero division
        normal_x /= normal_mag
        normal_y /= normal_mag

        # Incoming velocity
        vx, vy = puck_velocity

        # Compute dot product of velocity and normal
        dot_product = vx * normal_x + vy * normal_y

        # Compute reflected velocity using the elastic collision equation
        new_vx = vx - 2 * dot_product * normal_x
        new_vy = vy - 2 * dot_product * normal_y

        # Compute new angle in degrees
        new_angle = math.degrees(math.atan2(new_vy, new_vx))

        print(f"🔄 Corrected bounce angle: {new_angle:.2f}°")

        return new_angle



    def predict_puck_after_contact(self, mallet_position, puck_position):
        """
        Predicts the puck's new trajectory after it collides with the mallet.
        """
        impact_point = self.compute_contact_point(puck_position, mallet_position)
        print(f"\n--- Predicting impact at {mallet_position} ---")
        print(f"Mallet at: {mallet_position}")

        # Compute new bounce angle
        bounce_angle = self.compute_bounce_angle(puck_position, mallet_position)

        print(f"🎯 New bounce angle: {bounce_angle:.2f}°")

        # Simulate new trajectory after impact
        new_trajectory = self.predict_puck_trajectory(impact_point, bounce_angle)

        print(f"New trajectory after impact: {new_trajectory[:5]}")

        return new_trajectory

    def calculate_best_intercept(self, puck_trajectory):
        """
        Find the best intercept point and the best mallet center position 
        that leads the puck closest to the goal after impact.
        """
        goal_x, goal_y = self.table_width / 2, self.table_height  # Opponent's goal center
        best_intercept = None
        best_mallet_position = None
        best_distance_to_goal = float('inf')

        for step, (px, py) in enumerate(puck_trajectory):
            distance_to_puck = math.sqrt((px - self.mallet_position[0]) ** 2 + (py - self.mallet_position[1]) ** 2)
            effective_distance = max(0, distance_to_puck - (self.mallet_radius + self.puck_radius))
            time_to_reach = effective_distance / self.mallet_speed
            time_available = step * 0.01  # Time since start of trajectory

            if time_to_reach <= time_available:  # Mallet can reach in time
                # Get the puck's velocity at this step
                if step > 0:
                    prev_px, prev_py = puck_trajectory[step - 1]
                    puck_vx = px - prev_px
                    puck_vy = py - prev_py
                else:
                    puck_vx, puck_vy = 1e-3, 0  # Small nonzero default to prevent division by zero

                # Normalize the puck's velocity
                puck_speed = math.sqrt(puck_vx ** 2 + puck_vy ** 2) + 1e-6  # Prevent division by zero
                puck_vx /= puck_speed
                puck_vy /= puck_speed

                # **Compute the correct mallet placement by aligning with the tangent of the puck's velocity**
                # The mallet's center should be offset along the normal direction of the incoming trajectory

                # Compute normal vector (perpendicular to velocity)
                normal_x = -puck_vy  # Rotate by 90 degrees
                normal_y = puck_vx  

                # Compute mallet center so that the mallet's curvature interacts correctly with the puck
                best_mallet_x = px + normal_x * self.mallet_radius
                best_mallet_y = py + normal_y * self.mallet_radius

                # Ensure the mallet is inside the table bounds
                best_mallet_x = max(self.mallet_radius, min(self.table_width - self.mallet_radius, best_mallet_x))
                best_mallet_y = max(self.mallet_radius, min(self.table_height - self.mallet_radius, best_mallet_y))

                # Predict the puck's post-impact trajectory
                predicted_trajectory = self.predict_puck_after_contact((px, py), (best_mallet_x, best_mallet_y))

                if predicted_trajectory:
                    final_x, final_y = predicted_trajectory[-1]  # Last point of trajectory
                    distance_to_goal = math.sqrt((final_x - goal_x) ** 2 + (final_y - goal_y) ** 2)

                    # Update the best intercept if this leads the puck closer to the goal
                    if distance_to_goal < best_distance_to_goal:
                        best_intercept = (px, py)
                        best_mallet_position = (best_mallet_x, best_mallet_y)
                        best_distance_to_goal = distance_to_goal

        if best_intercept and best_mallet_position:
            print(f"🏆 Best intercept chosen at {best_intercept}, with mallet center at {best_mallet_position}, aiming for goal!")
        else:
            print("⚠️ No valid intercept found!")

        return best_intercept, best_mallet_position






    def visualize(self, puck_trajectory, mallet_position, intercept_point, predicted_trajectory):
        """
        Visualize the puck trajectory, mallet position, intercept position, and post-impact trajectory.
        """
        fig, ax = plt.subplots(figsize=(6, 12))

        # Plot original puck trajectory
        ax.plot(*zip(*puck_trajectory), '-b', label="Puck Trajectory Before Impact")
        ax.scatter(puck_trajectory[0][0], puck_trajectory[0][1], color='red', label="Starting Point")

        # Draw mallet as a circle (REAL RADIUS)
        mallet_circle = patches.Circle(mallet_position, self.mallet_radius, color='orange', alpha=0.5, label="Mallet")
        ax.add_patch(mallet_circle)

        # Draw the puck at the start of the trajectory (REAL RADIUS)
        puck_circle = patches.Circle(puck_trajectory[0], self.puck_radius, color='red', alpha=0.5, label="Puck")
        ax.add_patch(puck_circle)

        # Plot intercept point
        intercept_circle = patches.Circle(intercept_point, self.puck_radius, color="green", alpha=0.5, label="Intercept")
        ax.add_patch(intercept_circle)

        # Plot post-impact trajectory (starting from actual impact point)
        if predicted_trajectory:
            ax.plot(*zip(*predicted_trajectory), linestyle="dashed", label="After Impact")

        ax.set_xlim(0, self.table_width)
        ax.set_ylim(0, self.table_height)
        ax.legend()
        ax.set_title("Air Hockey Simulation with Proper Collision Visualization")
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        plt.show()


# Example usage:
ai = AirHockeyAI()

# Define initial puck position and impact point
import random
puck_position = (350, 1400)
trajectory = ai.predict_puck_trajectory(puck_position, random.uniform(-80, 80))

# Find the best intercept point
best_intercept, best_mallet_position = ai.calculate_best_intercept(trajectory)
print(f"Intercept at: {best_intercept}, Mallet should be at: {best_mallet_position}")

# If an intercept was found, move the mallet and visualize the interaction
if best_intercept:
    new_trajectory = ai.predict_puck_after_contact(best_intercept, puck_position)
    ai.visualize(trajectory, best_mallet_position, best_intercept, new_trajectory)
