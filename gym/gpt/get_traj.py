from datetime import datetime
import os
import gym
import numpy as np
import torch
import wandb
import json

import argparse
import pickle
import random
import sys
import copy

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(PROJECT_PATH)

# 파일 경로 설정
prompt_en_path = os.path.join(PROJECT_PATH, 'gpt/prompt_en.json')

def get_traj():
    # dataset_path = f'{PROJECT_PATH}/gym/data/combined_data.pkl'
    # dataset_path = f'/home/zzzanghun/git/decision-transformer/gym/data/medial/grid_4/ego-planner-data_1.pkl'
    dataset_path = '/home/zzzanghun/git/decision-transformer/gym/data/ego/ego-planner-data_90.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Define the indices of the actions to be used
    action_indices = [0, 1, 2, 6, 7, 8]
    obs_indices = [0, 1, 3, 4, 6, 7, 9, 10]
    
    for i in range(len(trajectories)):
        i = len(trajectories) - i - 1
        trajectories[i]['actions'] = trajectories[i]['actions'][:, action_indices]
        obs_first_part = trajectories[i]['observations'][:, :, :100*100]
        obs_second_part = trajectories[i]['observations'][:, :, 100*100:]
        obs_second_part = obs_second_part[:, :, obs_indices]
        trajectories[i]['observations'] = np.concatenate([obs_first_part, obs_second_part], axis=2)
        trajectories[i]['rewards'] = np.zeros(len(trajectories[i]['actions']), dtype=float)
        save_traj = False
        del_traj = False
        for j in range(len(trajectories[i]['actions'])):
            coef = trajectories[i]['actions'][j]
            # Discretize to 0.001 intervals
            coef = np.round(coef / 0.001) * 0.001
            # Assign back
            trajectories[i]['actions'][j] = coef
            a5, a4, a3, b5, b4, b3 = coef
            drone_info = trajectories[i]['observations'][j][:, 100*100:]
            v_x = drone_info[0][2]         # 현재 x축 속도
            v_y = drone_info[0][3]         # 현재 y축 속도
            a_x = drone_info[0][6]         # 현재 x축 가속도
            a_y = drone_info[0][7]         # 현재 y축 가속도
            direction_vector = trajectories[i]['observations'][j][:, 100*100:100*100 + 2]
            norm = np.linalg.norm(direction_vector)
            if norm != 0:
                direction_vector = direction_vector / norm
            trajectories[i]['observations'][j][:, 100*100:100*100 + 2] = direction_vector
            # 소수점 3째자리에서 반올림하고 문자열로 변환하여 출력
            rounded_values = np.round(trajectories[i]['observations'][j][:, 100*100:], 3)
            drone_info_str = str(rounded_values[0]).replace('\n', '').replace('  ', ' ')
            # print(rounded_values[0], "!!!!!!!!!")
            # 요소들을 쉼표로 구분하여 출력
            direction_to_target_str = ','.join([str(val) for val in rounded_values[0][:2]])
            cur_vel_str = ','.join([str(val) for val in rounded_values[0][2:4]])
            cur_acc_str = ','.join([str(val) for val in rounded_values[0][6:]])

            # print(direction_to_target_str)
            # print(cur_vel_str)
            # print(cur_acc_str)
            obs_observation = trajectories[i]['observations'][j][:, :100*100].reshape(100, 100)
            x0, y0 = 5, 5
            vx = []
            vy = []
            t_values = np.arange(0, 2.0 + 0.01, 0.01)
            for t in t_values:
                x = x0 + v_x * t + 0.5 * a_x * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                y = y0 + v_y * t + 0.5 * a_y * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5
                v_x_t = v_x + a_x * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
                v_y_t = v_y + a_y * t + 3 * b3 * t**2 + 4 * b4 * t**3 + 5 * b5 * t**4
                
                # grid map의 인덱스로 변환 (여기서는 반올림하여 정수 인덱스로 변환)
                ix = int(round(50 - (x - x0) * 10))
                iy = int(round(50 - (y - y0) * 10))
                
                # grid map의 범위 내에 있는 경우에만 값을 2로 지정
                if 0 <= ix < 100 and 0 <= iy < 100:
                    obs_observation[ix, iy] = 2  # 일반적으로 행이 y축, 열이 x축을 나타냄
                vx.append(v_x_t)
                vy.append(v_y_t)
            runlength_data = convert_to_runlength(obs_observation)
            print(obs_observation, "@!!@@!!@@!@!")

            vx_str = str([round(float(vx[i]), 3) for i in range(len(vx)) if i % 10 == 0]).replace(' ', '')
            vy_str = str([round(float(vy[i]), 3) for i in range(len(vy)) if i % 10 == 0]).replace(' ', '')

            prompt = update_prompt_en_json(vx_str, vy_str, runlength_data, cur_vel_str, cur_acc_str, direction_to_target_str)
            print(prompt, "@!!@@!!@@!@!")
# JSON 파일 직접 수정하는 부분 추가
def update_prompt_en_json(vx_data, vy_data, runlength_data, cur_vel_str, cur_acc_str, direction_to_target_str):
    # prompt_kor.json 파일 읽기
    with open(prompt_en_path, 'r') as f:
        prompt_data = json.load(f)
    
    # 속도 데이터 업데이트
    prompt_data['input_data']['drone_predicted_velocity']['predicted velocity']['data']['x_velocity'] = vx_data
    prompt_data['input_data']['drone_predicted_velocity']['predicted velocity']['data']['y_velocity'] = vy_data

    prompt_data['input_data']['grid_map']['run_length_data'] = runlength_data

    prompt_data['input_data']['drone_current_position_target_direction_and_state']['data']['current_velocity'] = cur_vel_str
    prompt_data['input_data']['drone_current_position_target_direction_and_state']['data']['current_acceleration'] = cur_acc_str
    prompt_data['input_data']['drone_current_position_target_direction_and_state']['data']['the target direction vector'] = direction_to_target_str

    return prompt_data

# 개선된 런랭스 변환 함수 - 2 값은 직선으로 처리
def convert_to_runlength(obs_observation):
    # 0을 제외한 고유 값들만 처리
    unique_values = [val for val in np.unique(obs_observation) if val != 0]
    runlength_data = []
    
    for value in unique_values:
        # 해당 값을 가진 위치 찾기
        positions = np.argwhere(obs_observation == value)
        
        if len(positions) == 0:
            continue
        
        # 값이 2인 경우 (궤적) - 직선으로 처리
        if value == 2:
            # 모든 점을 개별적으로 처리하거나 직선 세그먼트로 연결
            if len(positions) <= 2:
                # 점이 1-2개면 그냥 하나의 영역으로 처리
                min_row = np.min(positions[:, 0])
                max_row = np.max(positions[:, 0])
                min_col = np.min(positions[:, 1])
                max_col = np.max(positions[:, 1])
                
                runlength_data.append({
                    "rows": f"{min_row}-{max_row}",
                    "cols": f"{min_col}-{max_col}",
                    "value": "2"
                })
            else:
                # 점이 3개 이상이면 각 점을 개별적으로 처리
                for pos in positions:
                    row, col = pos
                    runlength_data.append({
                        "rows": f"{row}-{row}",
                        "cols": f"{col}-{col}",
                        "value": "2"
                    })
        else:
            # 값이 1인 경우 (장애물) - 기존 방식대로 처리
            # 행과 열 기준으로 정렬
            positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
            
            # 연속된 영역 찾기 (개선된 알고리즘)
            regions = []
            visited = set()
            
            for pos in positions:
                row, col = pos
                if (row, col) in visited:
                    continue
                    
                # 현재 위치에서 시작하는 영역 찾기
                min_row, max_row = row, row
                min_col, max_col = col, col
                queue = [(row, col)]
                visited.add((row, col))
                
                while queue:
                    r, c = queue.pop(0)
                    
                    # 4방향 탐색 (상하좌우)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited and 0 <= nr < obs_observation.shape[0] and 0 <= nc < obs_observation.shape[1]:
                            if obs_observation[nr, nc] == value:
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                                min_row = min(min_row, nr)
                                max_row = max(max_row, nr)
                                min_col = min(min_col, nc)
                                max_col = max(max_col, nc)
                
                # 영역 추가
                regions.append({
                    "rows": f"{min_row}-{max_row}",
                    "cols": f"{min_col}-{max_col}",
                    "value": str(int(value))
                })
            
            # 중복 제거 (동일한 영역 병합)
            merged_regions = []
            for region in regions:
                if not any(r["rows"] == region["rows"] and r["cols"] == region["cols"] and r["value"] == region["value"] for r in merged_regions):
                    merged_regions.append(region)
            
            runlength_data.extend(merged_regions)

    runlength_str = print_runlength_without_spaces_quotes(runlength_data)
    
    return runlength_str

# 런랭스 인코딩에서 원래 메트릭스로 복원하는 함수
def reconstruct_from_runlength(runlength_data, shape=(100, 100)):
    # 모든 요소가 0인 메트릭스 생성
    matrix = np.zeros(shape)
    
    # 런랭스 데이터를 사용하여 메트릭스 채우기
    for region in runlength_data:
        # 행과 열 범위 파싱
        row_range = region["rows"].split("-")
        col_range = region["cols"].split("-")
        
        # 시작과 끝 인덱스 추출
        row_start, row_end = int(row_range[0]), int(row_range[1])
        col_start, col_end = int(col_range[0]), int(col_range[1])
        
        # 값 추출
        value = float(region["value"])
        
        # 메트릭스 영역 채우기
        matrix[row_start:row_end+1, col_start:col_end+1] = value
    
    return matrix

# 띄어쓰기와 따옴표 없이 출력하는 함수
def print_runlength_without_spaces_quotes(runlength_data):
    result = "["
    for i, region in enumerate(runlength_data):
        if i > 0:
            result += ","
        result += "{rows:" + region["rows"] + ",cols:" + region["cols"] + ",value:" + region["value"] + "}"
    result += "]"
    return result

if __name__ == '__main__':
    get_traj()
