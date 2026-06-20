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