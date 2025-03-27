from airhocky_manual_ai_v31.AI_script_v31 import startup

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

def test_puck_trajectory(): #kinda dont know how test this function
    assert False


def test_calculate_intercept_point():
    assert False


def test_move_mallet_home():
    assert False


def test_defensive_action():
    assert False


def test_aggressive_action():
    assert False


def test_passive_aggressive_action():
    assert False
