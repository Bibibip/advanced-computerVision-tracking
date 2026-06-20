import math

def check_lost_status(state, cx, cy, current_sec_exact):
    just_dropped = False
    just_lost = False
    
    if state['owner_id'] is not None:
        if state['status'] == 'held':
            state['status'] = 'dropped'
            state['drop_time'] = current_sec_exact
            # 손에서 분리되는 순간의 좌표 저장
            state['drop_pos'] = (cx, cy) 
            just_dropped = True
            
        elif state['status'] == 'dropped':
            time_passed = current_sec_exact - state['drop_time']
            dist = math.hypot(cx - state['drop_pos'][0], cy - state['drop_pos'][1])
            
            # 💡 수정됨: 공중에서 바닥으로 떨어지는 거리를 고려해 거리 조건을 300으로 넉넉하게 완화
            # 또는 손에서 떨어진 지 3초가 지났다면 분실로 간주
            if time_passed >= 3.0 and dist < 300:
                state['status'] = 'lost'
                just_lost = True
                
    return just_dropped, just_lost

def check_recovered_status(state, new_overlapping_person, current_sec_exact):
    """dropped/lost 상태에서 누군가 다시 물건을 집어가면 '회수'로 판정.
    owner_id는 절대 바꾸지 않고, 최초 소유자 그대로 유지."""
    just_recovered = False
    is_owner_recovery = False
    
    if state['status'] in ('dropped', 'lost') and new_overlapping_person is not None:
        is_owner_recovery = (new_overlapping_person == state['owner_id'])
        state['status'] = 'held'
        state['recovered_time'] = current_sec_exact
        # ★ owner_id는 그대로 둠 (최초 소유자 고정)
        just_recovered = True
        
    return just_recovered, is_owner_recovery