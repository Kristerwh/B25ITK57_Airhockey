from rule_based_ai_agent_v31.AI_script_v31 import startup, TIME_STEP, TRAJECTORY_TIME_FRAME, MOVE_HOME_TICKS, MALLET_POS, \
    PUCK_SIZE, DEFENSIVE_ACTION_TICKS, ATTACK_SPREAD
import math

ai = startup()


def test_check_gooning():
    prev_gooning_counter = ai.get_gooning_counter()
    ai.set_mallet_pos((60,60))
    ai.check_gooning()
    assert prev_gooning_counter == (ai.get_gooning_counter() - 1), "should be equal since the puck is inside the gooning threshold"

    ai.set_gooning_counter(0)
    prev_gooning_counter = ai.get_gooning_counter()
    ai.set_mallet_pos((60,970))
    ai.check_gooning()
    assert prev_gooning_counter == (ai.get_gooning_counter() - 1), "should be equal since the puck is inside the gooning threshold"

    ai.set_gooning_counter(0)
    prev_gooning_counter = ai.get_gooning_counter()
    ai.set_mallet_pos((200, 970))
    ai.check_gooning()
    assert prev_gooning_counter != (ai.get_gooning_counter() - 1), "should not be equal since the puck is outside the gooning threshold"

    ai.set_gooning_counter(0)
    prev_gooning_counter = ai.get_gooning_counter()
    ai.set_mallet_pos((160, 60))
    ai.check_gooning()
    assert prev_gooning_counter != (ai.get_gooning_counter() - 1), "should not be equal since the puck is outside the gooning threshold"



def test_check_safe_to_move_home():
    p1 = (100,100)
    p2 = (110,110)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.check_safe_to_move_home() is True, "should return true since the puck is moving away from the ai side"

    p1 = (110, 110)
    p2 = (100, 100)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.check_safe_to_move_home() is None, "should return none since the puck is moving towards the ai side"

    p1 = (1000, 100)
    p2 = (1100, 100)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.check_safe_to_move_home() is True, "should return true since the puck is at the other side of the table"

    p1 = (1100, 100)
    p2 = (1000, 100)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.check_safe_to_move_home() is True, "should return true since the puck is at the other side of the table"


def test_update_positions():
    p1 = (100,100)
    p2 = (110,110)
    p3 = (120,120)

    ai.update_positions(p1)
    assert len(ai.puck_positions) == 2, "the array should not have more or less than 2 values/tuples"
    assert ai.puck_positions[0] and ai.puck_positions[1] == p1, "should load the same value twice if it only gets one value"

    ai.update_positions(p2)
    ai.update_positions(p3)

    assert len(ai.puck_positions) == 2, "the array should not have more or less than 2 values/tuples"
    assert ai.puck_positions[0] == p2 and ai.puck_positions[1] == p3, "these should be equal meaning that the array is correctly loaded with values"


def test_reset_all_ticks():
    ai.set_move_home_ticks(123)
    ai.set_defensive_action_ticks(123)
    ai.set_aggressive_action_ticks(123)
    ai.set_no_intercept_ticks(123)
    ai.set_passive_aggressive_action_ticks(123)

    ai.reset_all_ticks()
    assert ai.get_move_home_ticks() == 0, "should be 0 after reset"
    assert ai.get_defensive_action_ticks() == 0, "should be 0 after reset"
    assert ai.get_aggressive_action_ticks() == 0, "should be 0 after reset"
    assert ai.get_no_intercept_ticks() == 0, "should be 0 after reset"
    assert ai.get_passive_aggressive_action_ticks() == 0, "should be 0 after reset"


def test_calculate_velocity():
    p1 = (100,100)
    p2 = (101,101)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.calculate_velocity() == (1000,1000), "should be equal since the velocity is the same as the differance divided by the timestep (1000Hz)"

    p1 = (0,0)
    p2 = (0,0)
    ai.update_positions(p1)
    ai.update_positions(p2)
    assert ai.calculate_velocity() == (0,0),"should return 0,0 since there is no difference in the positions "

def test_puck_trajectory():
    puck_pos = (10, 10)
    puck_vel = (5, 3)

    trajectory, trajectory_time = ai.puck_trajectory(puck_pos, puck_vel)

    # Check if the length of trajectory and time arrays match expected iterations
    expected_steps = int(TRAJECTORY_TIME_FRAME/TIME_STEP)
    assert (len(trajectory), expected_steps)
    assert (len(trajectory_time), expected_steps)

    # Check if the trajectory starts at the given initial position
    assert math.isclose(trajectory[0][0], puck_pos[0] + puck_vel[0] * TIME_STEP, rel_tol=1e-2)
    assert math.isclose(trajectory[0][1], puck_pos[1] + puck_vel[1] * TIME_STEP, rel_tol=1e-2)

    # Check if the time values increase correctly
    for i in range(1, len(trajectory_time)):
        assert math.isclose(trajectory_time[i], trajectory_time[i - 1] + TIME_STEP, rel_tol=1e-5)


def test_calculate_intercept_point():

    # Case where puck enters defense box
    trajectory = [(670, 500), (660, 500), (650, 500), (640, 500)]
    trajectory_time = [0.1, 0.2, 0.3, 0.4]
    intercept, time = ai.calculate_intercept_point(trajectory, trajectory_time)
    assert intercept == (640, 500), "should be able to calculate the intercept point"
    assert time == 0.4

    # Case where puck does not enter defense box
    trajectory = [(670, 100), (660, 100), (650, 100), (640, 100)]
    trajectory_time = [0.1, 0.2, 0.3, 0.4]
    intercept, time = ai.calculate_intercept_point(trajectory, trajectory_time)
    assert intercept is None
    assert time is None


def test_move_mallet_home():
    #home pos is 100,519
    ai.set_mallet_pos((300, 500))  # Example current position

    vx, vy, ticks = ai.move_mallet_home()

    expected_vx = (MALLET_POS[0] - ai.mallet_pos[0]) / MOVE_HOME_TICKS
    expected_vy = (MALLET_POS[1] - ai.mallet_pos[1]) / MOVE_HOME_TICKS

    assert math.isclose(vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(vy, expected_vy, rel_tol=1e-5)
    assert ticks == MOVE_HOME_TICKS


def test_defensive_action():
    ai.set_mallet_pos((300, 500))  # Example current position

    # Case where a valid block position is found
    trajectory = [(670, 500), (660, 500), (650, 500), (640, 500)]
    vx, vy, ticks = ai.defensive_action(trajectory)
    expected_target = (640-PUCK_SIZE,500)
    expected_vx = (expected_target[0] - ai.get_mallet_pos()[0]) / DEFENSIVE_ACTION_TICKS
    expected_vy = (expected_target[1] - ai.get_mallet_pos()[1]) / DEFENSIVE_ACTION_TICKS

    assert math.isclose(vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(vy, expected_vy, rel_tol=1e-5)
    assert ticks == DEFENSIVE_ACTION_TICKS

    # Case where no valid block position is found
    trajectory = [(670, 100), (660, 100), (650, 100), (640, 100)]
    vx, vy, ticks = ai.defensive_action(trajectory)
    assert vx == 0
    assert vy == 0
    assert ticks == 0


def test_aggressive_action():
    #test for center hit
    ai.set_mallet_pos((300, 300))  # Example current position
    intercept_point = (640, 500)
    time_to_intercept = 0.11

    vx, vy, ticks = ai.aggressive_action(intercept_point, time_to_intercept)

    expected_ticks = round(time_to_intercept * 100)
    expected_target = (intercept_point[0] - PUCK_SIZE, intercept_point[1])
    expected_vx = (expected_target[0] - ai.get_mallet_pos()[0]) / expected_ticks
    expected_vy = (expected_target[1] - ai.get_mallet_pos()[1]) / expected_ticks

    assert math.isclose(vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(vy, expected_vy, rel_tol=1e-5)
    assert ticks == expected_ticks


    #test for buttom side hit
    ai.set_mallet_pos((300, 300))  # Example current position
    intercept_point = (640, 400)
    time_to_intercept = 0.11

    vx, vy, ticks = ai.aggressive_action(intercept_point, time_to_intercept)

    expected_ticks = round(time_to_intercept * 100)
    expected_target = (intercept_point[0] - PUCK_SIZE, intercept_point[1]-ATTACK_SPREAD)
    expected_vx = (expected_target[0] - ai.get_mallet_pos()[0]) / expected_ticks
    expected_vy = (expected_target[1] - ai.get_mallet_pos()[1]) / expected_ticks

    assert math.isclose(vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(vy, expected_vy, rel_tol=1e-5)
    assert ticks == expected_ticks

    #test for top side hit
    ai.set_mallet_pos((300, 300))  # Example current position
    intercept_point = (640, 650)
    time_to_intercept = 0.11

    vx, vy, ticks = ai.aggressive_action(intercept_point, time_to_intercept)

    expected_ticks = round(time_to_intercept * 100)
    expected_target = (intercept_point[0] - PUCK_SIZE, intercept_point[1] + ATTACK_SPREAD)
    expected_vx = (expected_target[0] - ai.get_mallet_pos()[0]) / expected_ticks
    expected_vy = (expected_target[1] - ai.get_mallet_pos()[1]) / expected_ticks

    assert math.isclose(vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(vy, expected_vy, rel_tol=1e-5)
    assert ticks == expected_ticks


def test_passive_aggressive_action(): #not finished
    assert True
