import math

def initialize_item_state():
    return {
        "owner_id": None,
        "status": "idle",

        "drop_time": 0.0,
        "drop_pos": (0, 0),

        "candidate_owner": None,
        "overlap_start": 0.0,

        "missing_start": None

    }


# def find_overlapping_person(
#     item_center,
#     current_persons,
#     margin=150
# ):
#     cx, cy = item_center

#     nearest_id = None
#     nearest_dist = float("inf")

#     for person_id, (px1, py1, px2, py2) in current_persons.items():

#         pcx = (px1 + px2) / 2
#         pcy = (py1 + py2) / 2

#         dist = math.sqrt(
#             (cx - pcx) ** 2 +
#             (cy - pcy) ** 2
#         )

#         if dist < nearest_dist:
#             nearest_dist = dist
#             nearest_id = person_id

#     if nearest_dist < margin:
#         return nearest_id

#     return None

def find_nearest_person(
    item_center,
    current_persons
):
    cx, cy = item_center

    nearest_id = None
    nearest_dist = float("inf")

    for person_id, (px1, py1, px2, py2) in current_persons.items():

        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        dist = math.sqrt(
            (cx - pcx)**2 +
            (cy - pcy)**2
        )

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id = person_id

    return nearest_id, nearest_dist


def update_owner_state(
    state,
    candidate_person,
    current_sec_exact
):

    if candidate_person is None:
        return False, state["owner_id"]

    if state["owner_id"] is None:

        state["owner_id"] = candidate_person
        state["status"] = "held"

        return True, candidate_person

    return False, state["owner_id"]