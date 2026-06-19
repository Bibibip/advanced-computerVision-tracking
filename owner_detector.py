def initialize_item_state():
    return {
        "owner_id": None,
        "status": "idle",

        "drop_time": 0.0,
        "drop_pos": (0, 0),

        "candidate_owner": None,
        "overlap_start": 0.0
    }


def find_overlapping_person(
    item_center,
    current_persons,
    margin=50
):
    cx, cy = item_center

    for person_id, (px1, py1, px2, py2) in current_persons.items():

        if (
            (px1 - margin) <= cx <= (px2 + margin)
            and
            (py1 - margin) <= cy <= (py2 + margin)
        ):
            return person_id

    return None


def update_owner_state(
    state,
    overlapping_person,
    current_sec_exact
):
    """
    Returns
    -------
    owner_confirmed : bool
    owner_id : int | None
    """

    if overlapping_person is None:
        state["candidate_owner"] = None
        return False, state["owner_id"]

    if state["owner_id"] is None:

        if state["candidate_owner"] != overlapping_person:

            state["candidate_owner"] = overlapping_person
            state["overlap_start"] = current_sec_exact

        else:

            overlap_duration = (
                current_sec_exact
                - state["overlap_start"]
            )

            if overlap_duration >= 2.0:

                state["owner_id"] = overlapping_person
                state["status"] = "held"

                return True, overlapping_person

    else:

        if state["owner_id"] == overlapping_person:
            state["status"] = "held"

    return False, state["owner_id"]