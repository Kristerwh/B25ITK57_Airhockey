import matplotlib.pyplot as plt
import numpy as np



MALLET_POS = 250,50
MALLET_SPEED = 1000
MALLET_SIZE = 10
PUCK_SIZE = 10
TABLE_WIDTH = 500
TABLE_HEIGHT = 1500
TIME_STEP = 0.01 #100hz
TRAJECTORY_TIME_FRAME = 1 #how long to predict the puck trajectory for in seconds
ATTACK_SPREAD = 30



class AirHockeyAI:
    def __init__(self,mallet_pos, mallet_speed, mallet_size, puck_size, table_width, table_height):

        self.mallet_pos = mallet_pos
        self.mallet_speed = mallet_speed
        self.mallet_size = mallet_size
        self.puck_pos = None
        self.puck_size = puck_size
        self.table_width = table_width
        self.table_height = table_height
        self.puck_positions = []


    def update_positions(self, new_pos):
        if len(self.puck_positions) >= 2:
            self.puck_positions.pop(0)
        self.puck_positions.append(new_pos)
        self.puck_pos = new_pos

    def get_positions(self):
        return self.puck_positions

    def move_mallet_home(self):
        self.mallet_pos = 250,100
        return self.mallet_pos


    def calculate_velocity(self, time_step):
        x1,y1 = self.puck_positions[0]
        x2,y2 = self.puck_positions[1]

        vx = (x2-x1)/time_step
        vy = (y2-y1)/time_step

        return vx,vy

    def puck_trajectory(self, puck_pos, puck_vel, time_frame):
        trajectory = []
        trajectory_time = []
        px,py = puck_pos
        vx,vy = puck_vel

        dt = TIME_STEP
        time_elapsed = 0

        while  time_elapsed < time_frame:
            px += vx*dt
            py += vy*dt

            #check if the puck hits the wall
            if px <= self.puck_size or px >= self.table_width - self.puck_size:
                vx = -vx
            if py <= self.puck_size or py >= self.table_height - self.puck_size:
                vy = -vy

            trajectory.append((px, py))
            trajectory_time.append(time_elapsed)
            time_elapsed += dt

        return trajectory, trajectory_time


    def calculate_intercept_point(self,trajectory,trajectory_time):
        idx = 0
        for px,py in trajectory:
            idx += 1
            if px > 100 and px < 400 and py < 400:
                intercept_point = (px,py)
                time_to_intercept = trajectory_time[idx]
                return intercept_point,time_to_intercept

    def defencive_action(self, trajectory):
        mallet_pos_tuple = self.mallet_pos #home pos is 250,100 (x,y)
        mallet_pos = np.array([mallet_pos_tuple[0],mallet_pos_tuple[1]])
        N = 10

        for px,py in trajectory:
            if px > 150 and px < 350 and py < 200:
                target = np.array([px,py])
                mallet_trajectory = [list(mallet_pos + (i / (N - 1)) * (target - mallet_pos)) for i in range(N)]
                return mallet_trajectory

    def aggressiv_action(self, intercept_point, time_to_intercept):
        px, py = intercept_point
        attack_posistions = []
        attack_posistions.append((px, py - ATTACK_SPREAD))
        attack_posistions.append((px, py))
        attack_posistions.append((px, py + ATTACK_SPREAD))








#plotting_functions
def plot_trajectory(trajectory):
    plt.figure(figsize=(6, 12))
    plt.plot(*zip(*trajectory), '-b', label="Puck Trajectory")
    plt.scatter(trajectory[0][0], trajectory[0][1], color='red', label="Starting Point")
    plt.xlim(0, TABLE_WIDTH)
    plt.ylim(0, TABLE_HEIGHT)
    plt.show()




ai = AirHockeyAI(MALLET_POS,MALLET_SPEED,MALLET_SIZE,PUCK_SIZE,TABLE_WIDTH,TABLE_HEIGHT)

p1 = (40,800)
p2 = (30,767)

ai.update_positions(p1)
ai.update_positions(p2)

puck_velocity = ai.calculate_velocity(TIME_STEP)
print(puck_velocity)

puck_trajectory, puck_trajectory_time = ai.puck_trajectory(ai.get_positions()[0],puck_velocity,TRAJECTORY_TIME_FRAME)
print(puck_trajectory)
plot_trajectory(puck_trajectory)

intercept_point, intercept_time = ai.calculate_intercept_point(puck_trajectory,puck_trajectory_time)

print(intercept_point , intercept_time)

mallet_trajectory = ai.defencive_action(puck_trajectory)
print (mallet_trajectory)

plot_trajectory(mallet_trajectory)