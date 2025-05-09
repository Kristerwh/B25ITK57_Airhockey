import numpy as np

#sizes
MALLET_SPEED = 1000 #not used
MALLET_SIZE = 100 #not used
PUCK_SIZE = 31.65
TABLE_WIDTH = 1948
TABLE_HEIGHT = 1038

#zones
DEFENSE_BOX_X = (TABLE_WIDTH/3) #649.333
DEFENSE_BOX_Y = (TABLE_HEIGHT/6), ((5 * TABLE_HEIGHT) / 6)# 173,865
GOONING_THRESHOLD = [(120,120),(120,TABLE_HEIGHT-120)]

#div
MALLET_POS = 100 ,(TABLE_HEIGHT / 2)
TIME_STEP = 0.001 #this is used for velocity and trajectory calculations, it does not need to be equal to the simulation timestep
TRAJECTORY_TIME_FRAME = 0.15 #how long to predict the puck trajectory for in seconds

#offsets
ATTACK_SPREAD = 50
PASSIVE_AGGRESSIVE_ACTION_OFFSET = 15
DEFENSIVE_ACTION_BOX_OFFSET = 0

#ticks and counter threshold
#these tick constants decides how long to wait or do certain actions
MOVE_HOME_TICKS = 5
DEFENSIVE_ACTION_TICKS = 10
PASSIVE_AGGRESSIVE_TICKS = 40
PASSIVE_AGGRESSIVE_DELAY_TICKS = 3000
MAXIMUM_ALLOWED_GOONING = 3000



#this class contains the necessary variables and functions for the AI script
class AirHockeyAI:
    def __init__(self, mallet_speed = MALLET_SPEED, mallet_size = MALLET_SIZE, puck_size = PUCK_SIZE, table_width = TABLE_WIDTH, table_height = TABLE_HEIGHT, time_step = TIME_STEP, trajectory_time_frame = TRAJECTORY_TIME_FRAME):
        self.mallet_pos = None
        self.mallet_speed = mallet_speed #not used
        self.mallet_size = mallet_size #not used
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
        self.gooning_counter = 0

    #different get functions
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

    def get_positions(self):
        return self.puck_positions

    def get_mallet_pos(self):
        return self.mallet_pos

    def get_gooning_counter(self):
        return self.gooning_counter

    def get_no_intercept_ticks(self):
        return self.no_intercept_ticks

    def get_time_step(self):
        return self.time_step

    #diffent set functions
    def set_mallet_pos(self, new_pos):
        self.mallet_pos = new_pos

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

    def set_gooning_counter(self,ticks):
        self.gooning_counter = ticks

    #  functions to check if the AI should do certain actions
    def check_gooning(self):
        #this function uses the gooning thresholds to detect if the puck is stuck in a corner of the table
        #if this detects that the puck is indeed stuck in the corner this function will make the mallet hit the puck repeatedly until its no longer stuck in the corner
        #without this function the AI will soft lock itself occasionally
        if (self.mallet_pos[0] < GOONING_THRESHOLD[0][0] and self.mallet_pos[1] < GOONING_THRESHOLD[0][1]) or (self.mallet_pos[0] < GOONING_THRESHOLD[1][0] and self.mallet_pos[1] > GOONING_THRESHOLD[1][1]):
            #this if-statement checks if the puck is in a corner position and the adds 1 to the counter
            self.gooning_counter += 1
            if self.gooning_counter >= MAXIMUM_ALLOWED_GOONING:
                #this if-statement checks if the puck have been in a corner position for a certain length of time and will force the mallet to reset itself to the home position
                self.gooning_counter = 0
                self.reset_all_ticks()
                mallet_vx, mallet_vy,ticks = self.move_mallet_home()
                self.set_mallet_vx(mallet_vx)
                self.set_mallet_vy(mallet_vy)
                self.set_move_home_ticks(ticks)
        else:
            #this else-statement makes sure that the counter resets if the mallet is no longer in the corner
            self.set_gooning_counter(0)

    def check_safe_to_move_home(self):
        #this function checks if it's safe for the mallet to move home, this determines this by looking the direction the puck is moving and looking at which side of the table the puck currently is at
        vx, _  = self.calculate_velocity()
        px, _ = self.puck_positions[1]
        if vx > 0 or px > TABLE_WIDTH/2:
            #if the puck is moving away or is at the opposite side of the table this returns true
            return True


    def update_positions(self, new_pos):
        #adds a new puck position to the puck_position array and to the puck_pos variable
        #it also makes sure the array is not longer than 2
        if len(self.puck_positions) >= 2:
            self.puck_positions.pop(0)
        self.puck_positions.append(new_pos)
        self.puck_pos = new_pos
        if len(self.puck_positions) < 2:
            self.update_positions(new_pos)

    def reset_all_ticks(self):
        #resets all ticks, this effectively interrupts current move cycle
        self.move_home_ticks = 0
        self.defensive_action_ticks = 0
        self.aggressive_action_ticks = 0
        self.no_intercept_ticks = 0
        self.passive_aggressive_action_ticks = 0


    def calculate_velocity(self):
        #this calculates the velocity of the puck using 2 puck positions and returns this value as a tuple
        #this data will be used by the puck_trajectory function
        time_step = self.time_step

        x1,y1 = self.puck_positions[0]
        x2,y2 = self.puck_positions[1]

        vx = (x2-x1)/time_step
        vy = (y2-y1)/time_step

        puck_vel = vx,vy
        return puck_vel

    def puck_trajectory(self, puck_pos, puck_vel):
        #todo optimize plz :)
        #calculate the pucks trajectory using the velocity of the puck and the last puck position
        #this returns 2 arrays one with the coordinates of each point and the other with the corresponding time for these points
        #this will be used by the calculate_intercept_point function
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

            if time_elapsed % 0.01 == 0:
                trajectory.append((px, py))
                trajectory_time.append(time_elapsed)
            time_elapsed += dt

        return trajectory, trajectory_time

    def calculate_intercept_point(self,trajectory,trajectory_time):
        #this checks if the puck will enter the AI's defense box
        #if it sees that it will enter the box the function will return the intercept point and the time for intercept
        #if the puck trajectory does not enter the box this function will return None,None
        idx = -1
        for px,py in trajectory:
            idx += 1
            if DEFENSE_BOX_Y[0] < py < DEFENSE_BOX_Y[1] and px < DEFENSE_BOX_X:
                intercept_point = (px,py)
                time_to_intercept = trajectory_time[idx]
                self.set_no_intercept_ticks(0)
                return intercept_point,time_to_intercept
        return None,None



    def move_mallet_home(self):
        #find the VX and VY for the mallet to move to home position form its current position over the specified number of ticks
        #returns VX, VY and the number of ticks this action takes
        mallet_pos = np.array([self.mallet_pos[0], self.mallet_pos[1]])
        target = np.array([MALLET_POS[0], MALLET_POS[1]])

        vx, vy = (target - mallet_pos) / MOVE_HOME_TICKS

        return vx, vy, MOVE_HOME_TICKS

    def defensive_action(self, trajectory):
        #moves the mallet form its current position to a position to block the incoming puck along the predicted trajectory
        #returns VX, VY and the number of ticks this action takes
        mallet_pos_tuple = self.mallet_pos
        mallet_pos = np.array([mallet_pos_tuple[0],mallet_pos_tuple[1]])

        for px,py in trajectory:
            if DEFENSE_BOX_Y[0]+DEFENSIVE_ACTION_BOX_OFFSET < py < DEFENSE_BOX_Y[1]-DEFENSIVE_ACTION_BOX_OFFSET and px < DEFENSE_BOX_X-DEFENSIVE_ACTION_BOX_OFFSET:
                target = np.array([px - PUCK_SIZE,py])

                vx, vy = (target - mallet_pos) / DEFENSIVE_ACTION_TICKS
                return vx, vy, DEFENSIVE_ACTION_TICKS

        return 0,0,0

    def aggressive_action(self, intercept_point, time_to_intercept):
        #is a more aggressive move to do inside the defensive box
        px, py = intercept_point
        mallet_pos_tuple = self.mallet_pos
        mallet_pos = np.array([mallet_pos_tuple[0], mallet_pos_tuple[1]])

        ticks = round(time_to_intercept * 100)

        if py <= (TABLE_HEIGHT / 2) - 100:
            target = (px - PUCK_SIZE, py - ATTACK_SPREAD)
        elif (TABLE_HEIGHT / 2) - 100 < py < (TABLE_HEIGHT / 2) + 100:
            target = (px - PUCK_SIZE, py)
        else:
            target =(px - PUCK_SIZE, py + ATTACK_SPREAD)

        vx,vy = (target - mallet_pos)/ ticks
        return vx,vy,ticks

    def passive_aggressive_action(self):
        #this is currently the AI default move
        #this function makes sure the AI will always get the puck over to the other side of the table
        px, py  = self.puck_positions[1]
        if px <= TABLE_WIDTH/2:
            self.no_intercept_ticks += 1
            if self.no_intercept_ticks >= PASSIVE_AGGRESSIVE_DELAY_TICKS:
                ticks = PASSIVE_AGGRESSIVE_TICKS
                self.set_passive_aggressive_action_ticks(ticks + 1)
                mallet_pos_tuple = self.mallet_pos
                mallet_pos = np.array([mallet_pos_tuple[0], mallet_pos_tuple[1]])
                if (px - PASSIVE_AGGRESSIVE_ACTION_OFFSET) <= 0:
                    px = PUCK_SIZE + PASSIVE_AGGRESSIVE_ACTION_OFFSET + 1
                target = (px - PASSIVE_AGGRESSIVE_ACTION_OFFSET, py)
                #target = (px - PASSIVE_AGGRESSIVE_ACTION_OFFSET, py + 1) use this if the ai script keeps getting stuck at the back wall without being in the corner, NOTE this is a very unlikely senaro but can cause a soft lock of the script
                vx, vy = (target - mallet_pos) / ticks
                return vx, vy
        else:
            self.set_no_intercept_ticks(0)
        return 0,0



#main functions
def startup():
    #uses this to initialise the AI script
    ai = AirHockeyAI()
    return ai

#will only return vx, vy for mallet
def run(ai, puck_pos, mallet_pos):
    #this is the main componet that makes the AI work, this contains the logic to deside what move to do when

    #updates the AI class with the current positions of the puck and mallet
    ai.update_positions(puck_pos)
    ai.set_mallet_pos(mallet_pos)

    #checks if the puck is stuck in a corner
    ai.check_gooning()

    #gets the current tick counters from the class for internal use
    move_home_ticks = ai.get_move_home_ticks()
    defensive_action_ticks = ai.get_defensive_action_ticks()
    aggressive_action_ticks = ai.get_aggressive_action_ticks()
    passive_aggressive_ticks = ai.get_passive_aggressive_action_ticks()

    #gets the last used velocity of the mallet. this is used for move execution
    mallet_vx = ai.get_mallet_vx()
    mallet_vy = ai.get_mallet_vy()

    #these if statements checks if the AI is currently is executing a move
    if passive_aggressive_ticks > 0:
        passive_aggressive_ticks -= 1
        ai.set_passive_aggressive_action_ticks(passive_aggressive_ticks)
        if passive_aggressive_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if move_home_ticks > 0:
        move_home_ticks -= 1
        ai.set_move_home_ticks(move_home_ticks)
        if move_home_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    if defensive_action_ticks > 0:
        defensive_action_ticks -= 1
        ai.set_defensive_action_ticks(defensive_action_ticks)
        if defensive_action_ticks == 0:
            mallet_vx, mallet_vy, ticks = ai.move_mallet_home()
            ai.set_move_home_ticks(ticks + 1)
            ai.set_mallet_vx(mallet_vx)
            ai.set_mallet_vy(mallet_vy)
            return mallet_vx, mallet_vy, ticks
        return mallet_vx, mallet_vy

    if aggressive_action_ticks > 0:
        aggressive_action_ticks -= 1
        ai.set_aggressive_action_ticks(aggressive_action_ticks)
        if aggressive_action_ticks == 0:
            return 0, 0
        return mallet_vx, mallet_vy

    #function call to calculate velocity, trajectory and intercept point
    puck_vel = ai.calculate_velocity()
    trajectory, trajectory_time = ai.puck_trajectory(ai.puck_positions[1], puck_vel)
    intercept_point, time_to_intercept  = ai.calculate_intercept_point(trajectory, trajectory_time)

    #if there is no intercept point, meaning the puck will not travel inside the defense box the AI will execute the passiv_aggressiv_action
    if intercept_point is None:
        mallet_vx, mallet_vy = ai.passive_aggressive_action()
        if ai.check_safe_to_move_home() and mallet_vx == 0 and mallet_vy == 0:
            mallet_vx, mallet_vy, ticks = ai.move_mallet_home()
            ai.set_move_home_ticks(ticks + 1)
            ai.set_mallet_vx(mallet_vx)
            ai.set_mallet_vy(mallet_vy)
            return mallet_vx, mallet_vy
        ai.set_mallet_vx(mallet_vx)
        ai.set_mallet_vy(mallet_vy)
        return mallet_vx, mallet_vy
    else:
        ai.set_no_intercept_ticks(0)

    #checks if the intercept time allows for an aggressiv move(more than 0.1 sec) inside the defense box, else it will do a defensive move
    if time_to_intercept < 0.1:
        mallet_vx, mallet_vy, ticks = ai.defensive_action(trajectory)
        ai.set_defensive_action_ticks(ticks + 1)
    else:
        mallet_vx, mallet_vy, ticks = ai.aggressive_action(intercept_point, time_to_intercept)
        ai.set_aggressive_action_ticks(ticks + 1)

    #updates the mallet velocity for the AI class for move execution
    ai.set_mallet_vx(mallet_vx)
    ai.set_mallet_vy(mallet_vy)

    #returns the current VX and VY values to the simulation environment
    return mallet_vx, mallet_vy
