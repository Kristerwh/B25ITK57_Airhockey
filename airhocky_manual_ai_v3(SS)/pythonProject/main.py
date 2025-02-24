import matplotlib.pyplot as plt
import numpy as np



MALLET_POS = 250,50
MALLET_SPEED = 1000
MALLET_SIZE = 10
PUCK_SIZE = 10
TABLE_WIDTH = 500
TABLE_HEIGHT = 1500
TIME_STEP = 0.01 #100hz
TRAJECTORY_TIME_FRAME = 0.2 #how long to predict the puck trajectory for in seconds
ATTACK_SPREAD = 30



class AirHockeyAI:
    def __init__(self,mallet_pos, mallet_speed, mallet_size, puck_size, table_width, table_height, time_step, trajectory_time_frame):
        self.mallet_pos = mallet_pos
        self.mallet_speed = mallet_speed
        self.mallet_size = mallet_size
        self.puck_pos = None
        self.puck_size = puck_size
        self.table_width = table_width
        self.table_height = table_height
        self.puck_positions = []
        self.time_step = time_step
        self.trajectory_time_frame = trajectory_time_frame


    def update_positions(self, new_pos):
        if len(self.puck_positions) >= 2:
            self.puck_positions.pop(0)
        self.puck_positions.append(new_pos)
        self.puck_pos = new_pos


    def get_positions(self):
        return self.puck_positions

    def get_mallet_pos(self):
        return self.mallet_pos


    def move_mallet_home(self): #TODO: implement a way determine if its safe to move home
        mallet_pos = np.array([self.mallet_pos[0], self.mallet_pos[1]])
        target = np.array([MALLET_POS[0], MALLET_POS[1]])
        N = 10 #how many ticks to move home

        mallet_trajectory = [list(mallet_pos + (i / (N - 1)) * (target - mallet_pos)) for i in range(N)]

        self.mallet_pos = MALLET_POS[0],MALLET_POS[1]
        return mallet_trajectory


    def check_safe_to_move_home(self):
        _,vy = self.calculate_velocity()
        if vy > 0:
            return True

    def calculate_velocity(self):
        time_step = self.time_step

        x1,y1 = self.puck_positions[0]
        x2,y2 = self.puck_positions[1]

        vx = (x2-x1)/time_step
        vy = (y2-y1)/time_step

        puck_vel = vx,vy
        return puck_vel


    def puck_trajectory(self, puck_pos, puck_vel):
        time_frame = self.trajectory_time_frame
        trajectory = []
        trajectory_time = []
        px,py = puck_pos
        vx,vy = puck_vel

        dt = self.time_step
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
            if 100 < px < 400 and py < 400:
                intercept_point = (px,py)
                time_to_intercept = trajectory_time[idx]
                return intercept_point,time_to_intercept
        return None,None


    def defencive_action(self, trajectory):
        mallet_pos_tuple = self.mallet_pos #home pos is 250,100 (x,y)
        mallet_pos = np.array([mallet_pos_tuple[0],mallet_pos_tuple[1]])
        N = 5

        for px,py in trajectory:
            if px > 150 and px < 350 and py < 200:
                target = np.array([px,py])
                self.mallet_pos = px, py
                mallet_trajectory = [list(mallet_pos + (i / (N - 1)) * (target - mallet_pos)) for i in range(N)]
                return mallet_trajectory


    def aggressiv_action(self, intercept_point, time_to_intercept): #TODO: finish
        px, py = intercept_point
        mallet_pos_tuple = self.mallet_pos  # home pos is 250,100 (x,y)
        mallet_pos = np.array([mallet_pos_tuple[0], mallet_pos_tuple[1]])

        N = round(time_to_intercept * 100)

        if px <=200:
            target = (px, py - ATTACK_SPREAD)
        elif px > 200 and px < 300:
            target = (px, py)
        else:
            target =(px, py + ATTACK_SPREAD)

        self.mallet_pos = target
        mallet_trajectory = [list(mallet_pos + (i / (N - 1)) * (target - mallet_pos)) for i in range(N)]
        return mallet_trajectory


#plotting_function
def plot_trajectory(trajectory):
    if len(trajectory) <= 2:
        return
    else:
        plt.figure(figsize=(6, 12))
        plt.plot(*zip(*trajectory), '-b', label="Puck Trajectory")
        plt.scatter(trajectory[0][0], trajectory[0][1], color='red', label="Starting Point")
        plt.xlim(0, TABLE_WIDTH)
        plt.ylim(0, TABLE_HEIGHT)
        plt.show()



#main functions
def startup():
    ai = AirHockeyAI(MALLET_POS, MALLET_SPEED, MALLET_SIZE, PUCK_SIZE, TABLE_WIDTH, TABLE_HEIGHT, TIME_STEP,TRAJECTORY_TIME_FRAME)
    ai.move_mallet_home()
    return ai

def run(new_pos):
    ai.update_positions(new_pos)

    if len(ai.puck_positions) <= 1:
        return
    else:
        puck_vel = ai.calculate_velocity()
        trajectory, trajectory_time = ai.puck_trajectory(ai.puck_positions[1], puck_vel)
        intercept_point, time_to_intercept  = ai.calculate_intercept_point(trajectory, trajectory_time)

        if intercept_point is None:
            print("No intercept point")
            return [ai.get_mallet_pos()]

        print("Time to intercept", time_to_intercept)

        if ai.check_safe_to_move_home():
            mallet_trajectory = ai.move_mallet_home()
            print("Moving Home")
        else:
            if time_to_intercept < 0.1:
                mallet_trajectory = ai.defencive_action(trajectory)
                print("Defencive Action")
            else:
                mallet_trajectory = ai.aggressiv_action(intercept_point, time_to_intercept)
                print("Aggressive Action")

        return mallet_trajectory

ai = startup()

p1 = (400,700)
p2 = (370,680)


run(p1)
mallet = run(p2)

print(mallet)
plot_trajectory(mallet)



