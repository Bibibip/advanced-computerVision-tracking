from utils import get_area_direction


def analyze_person_direction(
    person_paths,
    frame_width,
    frame_height
):

    final_direction = "변화 없음"

    if not person_paths:
        return final_direction

    main_person_id = max(
        person_paths.keys(),
        key=lambda k: len(person_paths[k])
    )

    path = person_paths[main_person_id]

    if len(path) < 2:
        return final_direction

    start_x, start_y = path[0]
    end_x, end_y = path[-1]

    start_area = get_area_direction(
        start_x,
        start_y,
        frame_width,
        frame_height
    )

    end_area = get_area_direction(
        end_x,
        end_y,
        frame_width,
        frame_height
    )

    if start_area == end_area:
        final_direction = f"{start_area} 머무름"
    else:
        final_direction = f"{start_area} ➔ {end_area}"

    return final_direction


def generate_report(
    detected_objects,
    final_direction,
    max_conf
):

    custom_detected = [
        x for x in detected_objects
        if x != "person"
    ]

    detected_item = (
        ", ".join(custom_detected).upper()
        if custom_detected
        else "사람 감지"
    )

    confidence = (
        f"{int(max_conf * 100)}%"
        if max_conf > 0
        else "0%"
    )

    return detected_item, final_direction, confidence