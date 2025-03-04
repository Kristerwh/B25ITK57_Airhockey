import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np




MALLET_SPEED = 1000
MALLET_SIZE = 100
PUCK_SIZE = 63.3
TABLE_WIDTH = 1948
TABLE_HEIGHT = 1038
MALLET_POS = 100 ,(TABLE_HEIGHT / 2)
DEFENSE_BOX_X = (TABLE_WIDTH/3)
DEFENSE_BOX_Y = (TABLE_HEIGHT/6), ((5 * TABLE_HEIGHT) / 6)
DEFENSIVE_ACTION_BOX_OFFSET = 50
TIME_STEP = 0.01 #100hz
TRAJECTORY_TIME_FRAME = 0.2 #how long to predict the puck trajectory for in seconds
ATTACK_SPREAD = 15
MOVE_HOME_TICKS = 5
DEFENSIVE_ACTION_TICKS = 5
PASSIVE_AGGRESSIVE_TICKS = 40




class AirHockeyAI:
    def __init__(self,mallet_pos = MALLET_POS, mallet_speed = MALLET_SPEED, mallet_size = MALLET_SIZE, puck_size = PUCK_SIZE, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT, time_step = TIME_STEP, trajectory_time_frame = TRAJECTORY_TIME_FRAME):
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
        self.move_home_ticks = 0
        self.defensive_action_ticks = 0
        self.aggressive_action_ticks = 0
        self.no_intercept_ticks = 0
        self.passive_aggressive_action_ticks = 0
        self.mallet_vx = 0
        self.mallet_vy = 0


    def get_move_home_ticks(self):
        return self.move_home_ticks

    def get_defensive_action_ticks(self):
        return self.defensive_action_ticks

    def get_aggressive_action_ticks(self):
        return self.aggressive_action_ticks

    def get_mallet_vx(self):
        return self.mallet_vx

    def get_mallet_vy(self):
        return self.mallet_vy

    def get_passive_aggressive_action_ticks(self):
        return self.passive_aggressive_action_ticks

    def set_passive_aggressive_action_ticks(self, ticks):
        self.passive_aggressive_action_ticks = ticks

    def set_no_intercept_ticks(self, ticks):
        self.no_intercept_ticks = ticks

    def set_move_home_ticks(self, ticks):
        self.move_home_ticks = ticks

    def set_defensive_action_ticks(self, ticks):
        self.defensive_action_ticks = ticks

    def set_aggressive_action_ticks(self, ticks):
        self.aggressive_action_ticks = ticks

    def set_mallet_vx(self, vx):
        self.mallet_vx = vx

    def set_mallet_vy(self, vy):
        self.mallet_vy = vy

    def update_positions(self, new_pos):
        if len(self.puck_positions) >= 2:
            self.puck_positions.pop(0)
        self.puck_positions.append(new_pos)
        self.puck_pos = new_pos

    def set_mallet_pos(self, new_pos):
        self.mallet_pos = new_pos

    def get_positions(self):
        return self.puck_positions

    def get_mallet_pos(self):
        return self.mallet_pos


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
        idx = -1
        for px,py in trajectory:
            idx += 1
            if DEFENSE_BOX_Y[0] < py < DEFENSE_BOX_Y[1] and px < DEFENSE_BOX_X:
                intercept_point = (px,py)
                time_to_intercept = trajectory_time[idx]
                return intercept_point,time_to_intercept
        return None,None


    def check_safe_to_move_home(self):
        vx, _  = self.calculate_velocity()
        if vx > 0:
            return True


    def move_mallet_home(self):
        mallet_pos = np.array([self.mallet_pos[0], self.mallet_pos[1]])
        target = np.array([MALLET_POS[0], MALLET_POS[1]])

        vx, vy = (target - mallet_pos) / MOVE_HOME_TICKS

        return vx, vy, MOVE_HOME_TICKS


    def defensive_action(self, trajectory):
        mallet_pos_tuple = self.mallet_pos
        mallet_pos = np.array([mallet_pos_tuple[0],mallet_pos_tuple[1]])

        for px,py in trajectory:
            if DEFENSE_BOX_Y[0]-DEFENSIVE_ACTION_BOX_OFFSET < py < DEFENSE_BOX_Y[1]-DEFENSIVE_ACTION_BOX_OFFSET and px < DEFENSE_BOX_X-DEFENSIVE_ACTION_BOX_OFFSET:
                target = np.array([px,py])

                vx, vy = (target - mallet_pos) / DEFENSIVE_ACTION_TICKS
                return vx, vy, DEFENSIVE_ACTION_TICKS

        return 0,0,0

    def passive_aggressive_action(self):
        self.no_intercept_ticks += 1
        if self.no_intercept_ticks >= 100:
            px, py  = self.puck_positions[1]
            if px <= TABLE_WIDTH/2:
                ticks = PASSIVE_AGGRESSIVE_TICKS
                mallet_pos_tuple = self.mallet_pos
                mallet_pos = np.array([mallet_pos_tuple[0], mallet_pos_tuple[1]])
                target = (px, py)
                vx, vy = (target - mallet_pos) / ticks
                return vx, vy, ticks


    def aggressive_action(self, intercept_point, time_to_intercept):
        px, py = intercept_point
        mallet_pos_tuple = self.mallet_pos
        mallet_pos = np.array([mallet_pos_tuple[0], mallet_pos_tuple[1]])

        ticks = round(time_to_intercept * 100)

        if py <= (TABLE_HEIGHT / 2) - 100:
            target = (px - ATTACK_SPREAD, py)
        elif (TABLE_HEIGHT / 2) - 100 < py < (TABLE_HEIGHT / 2) + 100:
            target = (px, py)
        else:
            target =(px + ATTACK_SPREAD, py)

        vx,vy = (target - mallet_pos)/ ticks
        return vx,vy,ticks




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
    ai = AirHockeyAI()
    return ai

#will only return vx, vy for mallet
def run(ai, puck_pos, mallet_pos):
    ai.update_positions(puck_pos)
    ai.set_mallet_pos(mallet_pos)

    move_home_ticks = ai.get_move_home_ticks()
    defensive_action_ticks = ai.get_defensive_action_ticks()
    aggressive_action_ticks = ai.get_aggressive_action_ticks()
    passive_aggressive_ticks = ai.get_passive_aggressive_ticks()
    mallet_vx = ai.get_mallet_vx()
    mallet_vy = ai.get_mallet_vy()


    if move_home_ticks > 0:
        print("Moving home ticks" , move_home_ticks)
        move_home_ticks -= 1
        ai.set_move_home_ticks(move_home_ticks)
        if move_home_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if defensive_action_ticks > 0:
        print("Defensive action ticks" , defensive_action_ticks)
        defensive_action_ticks -= 1
        ai.set_defensive_action_ticks(defensive_action_ticks)
        if defensive_action_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if aggressive_action_ticks > 0:
        print("Aggressive action ticks" , aggressive_action_ticks)
        aggressive_action_ticks -= 1
        ai.set_aggressive_action_ticks(aggressive_action_ticks)
        if aggressive_action_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if passive_aggressive_ticks > 0:
        print("Passive aggressive ticks" , passive_aggressive_ticks)
        passive_aggressive_ticks -= 1
        ai.set_passive_aggressive_ticks(passive_aggressive_ticks)
        if passive_aggressive_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if len(ai.puck_positions) <= 1:
        print("not enough puck positions")
        return 0, 0

    puck_vel = ai.calculate_velocity()
    trajectory, trajectory_time = ai.puck_trajectory(ai.puck_positions[1], puck_vel)
    intercept_point, time_to_intercept  = ai.calculate_intercept_point(trajectory, trajectory_time)

    if intercept_point is None:
        print("No intercept point")
        ai.get_passive_aggressive_action()
        if ai.check_safe_to_move_home():
            mallet_vx, mallet_vy, ticks = ai.move_mallet_home()
            print("Moving Home")
            ai.set_move_home_ticks(ticks + 1)
            ai.set_mallet_vx(mallet_vx)
            ai.set_mallet_vy(mallet_vy)
            return mallet_vx, mallet_vy
        return 0, 0

    else:
        ai.set_no_intercept_ticks(0)

    print("Time to intercept", time_to_intercept)
    if time_to_intercept < 0.1:
        mallet_vx, mallet_vy, ticks = ai.defensive_action(trajectory)
        ai.set_defensive_action_ticks(ticks + 1)
        print("Defencive Action")
    else:
        mallet_vx, mallet_vy, ticks = ai.aggressive_action(intercept_point, time_to_intercept)
        ai.set_aggressive_action_ticks(ticks + 1)
        print("Aggressive Action")

    ai.set_mallet_vx(mallet_vx)
    ai.set_mallet_vy(mallet_vy)
    return mallet_vx, mallet_vy
