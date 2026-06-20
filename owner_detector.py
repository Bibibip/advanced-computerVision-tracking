def find_overlapping_person(cx, cy, current_persons, margin=80):
    for p_tid, (px1, py1, px2, py2) in current_persons.items():
        if (px1 - margin) <= cx <= (px2 + margin) and (py1 - margin) <= cy <= (py2 + margin):
            return p_tid
    return None

# ★ 기존 함수 유지 + is_new_item 파라미터 추가 (기본값 False로 기존 로직 영향 없음)
def update_ownership(state, overlapping_person, current_sec_exact, is_new_item=False):
    just_owned = False
    
    # [추가된 로직] 물건이 처음 등장했을 때 겹친 사람이 있다면 즉시 소유자 할당
    if is_new_item and overlapping_person is not None:
        state['owner_id'] = overlapping_person
        state['status'] = 'held'
        return True 

    # [기존 로직 유지] 0.5초 대기 후 소유자 확정 로직
    if overlapping_person is not None:
        if state['owner_id'] is None:
            if state['candidate_owner'] != overlapping_person:
                state['candidate_owner'] = overlapping_person
                state['overlap_start'] = current_sec_exact
            else:
                if (current_sec_exact - state['overlap_start']) >= 0.5:
                    state['owner_id'] = overlapping_person
                    state['status'] = 'held'
                    just_owned = True
        else:
            if state['owner_id'] == overlapping_person:
                state['status'] = 'held'
    else:
        state['candidate_owner'] = None
        
    return just_owned