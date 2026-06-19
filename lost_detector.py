import math


def handle_drop_event(
    state,
    current_time,
    current_pos
):
    """
    물건이 사람과 분리된 순간 호출
    """

    state["status"] = "dropped"
    state["drop_time"] = current_time
    state["drop_pos"] = current_pos

    return state


def check_lost_condition(
    state,
    current_time,
    current_pos,
    time_threshold=3.0,
    distance_threshold=50
):
    """
    dropped 상태에서
    분실 여부 판단
    """

    if state["status"] != "dropped":
        return False

    elapsed = current_time - state["drop_time"]

    distance = math.hypot(
        current_pos[0] - state["drop_pos"][0],
        current_pos[1] - state["drop_pos"][1]
    )

    if (
        elapsed >= time_threshold
        and
        distance < distance_threshold
    ):
        state["status"] = "lost"
        return True

    return False


def check_retrieved_condition(
    state,
    overlapping_person
):
    """
    떨어진 물건을 다시 주웠는지 판정
    """

    if state["status"] != "dropped":
        return False

    if overlapping_person == state["owner_id"]:
        state["status"] = "retrieved"
        return True

    return False