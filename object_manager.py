import math

def match_items(current_items, item_states):
    """물건의 ID를 유지하고 매칭하는 로직"""
    matched_ids = set()
    for item in current_items:
        best_id = None
        # 떨어지는 물건의 빠른 속도를 감당하기 위해 탐색 반경을 400으로 설정
        best_dist = 400 

        for my_id, state in item_states.items():
            if state['name'] == item['name'] and my_id not in matched_ids:
                dist = math.hypot(item['cx'] - state['cx'], item['cy'] - state['cy'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = my_id

        if best_id is not None:
            item_states[best_id].update({'cx': item['cx'], 'cy': item['cy'], 'box': item['box']})
            matched_ids.add(best_id)
        else:
            new_id = len(item_states) + 1
            item_states[new_id] = {
                'name': item['name'], 'cx': item['cx'], 'cy': item['cy'], 'box': item['box'],
                'owner_id': None, 'status': 'idle', 'drop_time': 0.0, 'drop_pos': (item['cx'], item['cy']),
                'candidate_owner': None, 'overlap_start': 0.0
            }
            matched_ids.add(new_id)
    return matched_ids