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

# openai 라이브러리가 필요합니다.
import openai

import tiktoken

# OpenAI API 키 설정 (환경 변수로 관리 권장)
openai.api_key = ""

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(PROJECT_PATH)

# 파일 경로 설정
prompt_en_path = os.path.join(PROJECT_PATH, 'gpt/prompt_en.json')


def num_tokens_from_messages(messages, model="o3-mini"):
    """
    주어진 messages (ChatCompletion 포맷)를 model에 맞게 토크나이징하여,
    예상되는 총 토큰 수를 반환합니다.

    - OpenAI의 공식 예시( https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb )
      를 참조하여 만든 함수입니다.
    - gpt-3.5-turbo, gpt-4 등 ChatCompletion 모델마다 토크나이징 규칙이 조금씩 다를 수 있으므로
      tiktoken 라이브러리에서 해당 모델에 맞는 encoding을 사용합니다.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 모델별 구체적인 encoding이 없으면 기본 encodings을 가져옵니다. (ex. 모델 이름이 custom일 때)
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # 공식적인 ChatCompletion token counting 방식에 맞춰 처리
    # system, user, assistant 각각 메시지에 특별 토큰이 추가된다고 알려져 있습니다.
    # 아래 로직은 openai-cookbook의 예시를 그대로 따라갑니다.

    tokens_per_message = 4  # role + 두 개의 구분 토큰 + 기타
    tokens_per_name = -1    # name 필드가 있을 경우 보정

    num_tokens = 0
    for msg in messages:
        # 기본 4토큰(메시지 1개당) + content 토큰 수
        num_tokens += tokens_per_message
        for key, value in msg.items():
            # 예: "role", "content", "name" 등이 있을 수 있음
            num_tokens += len(encoding.encode(value))
            # 만약 name 필드가 존재하면 추가 보정
            if key == "name":
                num_tokens += tokens_per_name
    # 마지막에 시스템적으로 추가되는 종료 토큰(assistant가 답변을 생성하기 시작함을 의미)을 위해 2토큰
    num_tokens += 2
    return num_tokens

# --------------------------------------------------------------------------------
# ChatGPT(또는 openAI ChatCompletion) 호출하여 reward를 얻어오는 함수 예시
# --------------------------------------------------------------------------------
def get_reward_from_chat_gpt(prompt_data, episode_buffer):
    """
    prompt_data: JSON 형태의 prompt (파이썬 dict)

    실제로는 아래 예시처럼 ChatCompletion.create로 API를 호출하고,
    ChatGPT 응답을 원하는 reward 형식으로 파싱해야 합니다.
    """
    try:
        # 문자열 형태로 변환 (필요하다면)
        prompt_str = json.dumps(prompt_data, ensure_ascii=False)
        # messages=[
        #         {"role": "system", "content": "You are a judge evaluating the safety and efficiency of polynomial trajectories. Based on the provided information, assess the drone's trajectory and output only a single float value between 0 and 1. Do not include any explanations or additional text - provide only the numeric score."},
        #         {"role": "user", "content": prompt_str}
        #     ]
        # num_tokens = num_tokens_from_messages(messages, model="o3-mini")

        # 실제 ChatCompletion API 호출
        response = openai.ChatCompletion.create(
            model="o3-mini",  # 실제 사용 가능한 모델 명으로 바꾸세요. 예: "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You will assign rewards based on the given information. Carefully analyze all the data provided by the user and assign rewards according to the requirements stated in the user's question. To ensure data consistency, first generate 10 candidate rewards yourself and then calculate their average as the final reward."},
                {"role": "user", "content": prompt_str}
            ],
            reasoning_effort="low",
            seed=42        
        )

        # response 구조 예시: 
        # {
        #   'choices': [
        #       {
        #           'message': {
        #               'role': 'assistant',
        #               'content': "이 구간에 대한 reward는 0.54 입니다."
        #           },
        #           ...
        #       }
        #   ],
        #   ...
        # }
        content = response["choices"][0]["message"]["content"]

        # 임의로, 응답에서 float 숫자 하나를 찾아 reward로 사용한다고 가정
        # 실제로는 응답 포맷을 어떻게 할지에 따라 파싱 로직 달라집니다.
        reward_value = extract_float_from_string(content, episode_buffer)

        # 혹은 "reward: X"와 같은 특정 형식으로 응답하도록 프롬프트를 유도한다면
        # 정규표현식 등으로 파싱할 수 있습니다.

        return reward_value

    except Exception as e:
        print(f"[ERROR] ChatGPT API call failed: {str(e)}")
        if len(episode_buffer) > 0:
            save_path = f"/home/zzzanghun/git/decision-transformer/gym/data/GPT_reward/gtp_reward_data_ValueError.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(episode_buffer, f)
            print("saved episoed lenght: ", len(episode_buffer))
        sys.exit(1)


def extract_float_from_string(text: str, episode_buffer) -> float:
    """
    문자열 자체가 실수 하나만 포함되어 있다고 가정하고 float 변환.
    """
    try:
        return float(text.strip())
    except ValueError:
        print(f"[ERROR] 실수 변환 실패: '{text}'")
        print("프로세스를 종료합니다.")
        if len(episode_buffer) > 0:
            save_path = f"/home/zzzanghun/git/decision-transformer/gym/data/GPT_reward/gtp_reward_data_ValueError.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(episode_buffer, f)
            print("saved episoed lenght: ", len(episode_buffer))
        sys.exit(1)

# --------------------------------------------------------------------------------
# JSON 파일 직접 수정하는 부분
# --------------------------------------------------------------------------------
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
    unique_values = [val for val in np.unique(obs_observation) if val != 0]
    runlength_data = []
    
    for value in unique_values:
        positions = np.argwhere(obs_observation == value)
        if len(positions) == 0:
            continue
        
        # 값이 2인 경우 (궤적) - 직선으로 처리
        if value == 2:
            if len(positions) <= 2:
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
            positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
            regions = []
            visited = set()
            
            for pos in positions:
                row, col = pos
                if (row, col) in visited:
                    continue
                    
                min_row, max_row = row, row
                min_col, max_col = col, col
                queue = [(row, col)]
                visited.add((row, col))
                
                while queue:
                    r, c = queue.pop(0)
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
                
                regions.append({
                    "rows": f"{min_row}-{max_row}",
                    "cols": f"{min_col}-{max_col}",
                    "value": str(int(value))
                })
            
            merged_regions = []
            for region in regions:
                if not any(r["rows"] == region["rows"] and r["cols"] == region["cols"] and r["value"] == region["value"] for r in merged_regions):
                    merged_regions.append(region)
            
            runlength_data.extend(merged_regions)

    runlength_str = print_runlength_without_spaces_quotes(runlength_data)
    return runlength_str

def reconstruct_from_runlength(runlength_data, shape=(100, 100)):
    matrix = np.zeros(shape)
    for region in runlength_data:
        row_range = region["rows"].split("-")
        col_range = region["cols"].split("-")
        
        row_start, row_end = int(row_range[0]), int(row_range[1])
        col_start, col_end = int(col_range[0]), int(col_range[1])
        value = float(region["value"])
        
        matrix[row_start:row_end+1, col_start:col_end+1] = value
    
    return matrix

def print_runlength_without_spaces_quotes(runlength_data):
    result = "["
    for i, region in enumerate(runlength_data):
        if i > 0:
            result += ","
        result += "{rows:" + region["rows"] + ",cols:" + region["cols"] + ",value:" + region["value"] + "}"
    result += "]"
    return result


# --------------------------------------------------------------------------------
# get_traj() 함수 - Trajectory 받아와서 ChatGPT로 reward 생성 및 저장
# --------------------------------------------------------------------------------
def get_traj():
    dataset_path = f'{PROJECT_PATH}/data/combined_data.pkl'
    # dataset_path = f'{PROJECT_PATH}/data/medial/grid_4/ego-planner-data_1.pkl'
    # dataset_path = f'{PROJECT_PATH}/data/ego/ego-planner-data_90.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Define the indices of the actions to be used
    action_indices = [0, 1, 2, 6, 7, 8]
    obs_indices = [0, 1, 3, 4, 6, 7, 9, 10]
    
    # ChatGPT 호출 제한 및 pkl 저장 관련 변수
    gpt_call_count = 0
    gpt_call_limit = 10000  # 10000번 이상이면 중단

    # 에피소드 50개 단위로 저장
    episode_buffer = []
    save_count = 0

    random_seed = 42  # 원하는 시드 값 (아무 정수나 사용 가능)
    random.seed(random_seed)

    num_episodes = len(trajectories)
    episode_indices = list(range(num_episodes))
    random.shuffle(episode_indices)  # 인덱스 랜덤하게 섞기

    for i in episode_indices:
        trajectories[i]['actions'] = trajectories[i]['actions'][:, action_indices]

        obs_first_part = trajectories[i]['observations'][:, :, :100*100]
        obs_second_part = trajectories[i]['observations'][:, :, 100*100:]
        obs_second_part = obs_second_part[:, :, obs_indices]
        trajectories[i]['observations'] = np.concatenate([obs_first_part, obs_second_part], axis=2)

        # reward 초기화 (모두 0)
        trajectories[i]['rewards'] = np.zeros(len(trajectories[i]['actions']), dtype=float)

        # 각 step에 대해 ChatGPT 호출
        for j in range(len(trajectories[i]['actions'])):
            coef = trajectories[i]['actions'][j]
            coef = np.round(coef / 0.001) * 0.001
            trajectories[i]['actions'][j] = coef

            a5, a4, a3, b5, b4, b3 = coef

            drone_info = trajectories[i]['observations'][j][:, 100*100:]
            v_x = drone_info[0][2]         # 현재 x축 속도
            v_y = drone_info[0][3]         # 현재 y축 속도
            a_x = drone_info[0][6]         # 현재 x축 가속도
            a_y = drone_info[0][7]         # 현재 y축 가속도

            # 목표방향 정규화
            direction_vector = trajectories[i]['observations'][j][:, 100*100:100*100 + 2]
            norm = np.linalg.norm(direction_vector)
            if norm != 0:
                direction_vector = direction_vector / norm
            trajectories[i]['observations'][j][:, 100*100:100*100 + 2] = direction_vector

            # 소수점 3째자리에서 반올림한 값 문자열로
            rounded_values = np.round(trajectories[i]['observations'][j][:, 100*100:], 3)
            drone_info_str = str(rounded_values[0]).replace('\n', '').replace('  ', ' ')

            direction_to_target_str = ','.join([str(val) for val in rounded_values[0][:2]])
            cur_vel_str = ','.join([str(val) for val in rounded_values[0][2:4]])
            cur_acc_str = ','.join([str(val) for val in rounded_values[0][6:]])

            # Run-length 인코딩 위한 grid map
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

                ix = int(round(50 - (x - x0) * 10))
                iy = int(round(50 - (y - y0) * 10))
                if 0 <= ix < 100 and 0 <= iy < 100:
                    obs_observation[ix, iy] = 2
                vx.append(v_x_t)
                vy.append(v_y_t)

            runlength_data = convert_to_runlength(obs_observation)

            vx_str = str([round(float(vx[i]), 3) for i in range(len(vx)) if i % 10 == 0]).replace(' ', '')
            vy_str = str([round(float(vy[i]), 3) for i in range(len(vy)) if i % 10 == 0]).replace(' ', '')

            # prompt JSON 업데이트
            prompt = update_prompt_en_json(
                vx_str, vy_str, 
                runlength_data, 
                cur_vel_str, 
                cur_acc_str, 
                direction_to_target_str
            )
            # print(obs_observation, "@!!@@!!@@!@!")
            # print(prompt, "@!!@@!!@@!@!")

            # ----------------------
            # ChatGPT API 호출
            # ----------------------
            reward_value = get_reward_from_chat_gpt(prompt, episode_buffer)
            trajectories[i]['rewards'][j] = reward_value

            print(obs_observation)

            # GPT 호출 횟수 증가
            gpt_call_count += 1

            print(f"gpt_call_count: {gpt_call_count}")
            print(f"reward_value: {reward_value}")

        # 한 에피소드(trajectory) 완료 후 buffer에 추가
        episode_buffer.append(trajectories[i])

        # 50개 에피소드마다 저장
        if len(episode_buffer) == 50:
            save_count += 1
            save_path = f"/home/zzzanghun/git/decision-transformer/gym/data/GPT_reward/gtp_reward_data_{save_count}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(episode_buffer, f)
            print(f"에피소드 50개 저장 완료 -> {save_path}")
            # 버퍼 초기화
            episode_buffer = []

        # 모든 에피소드 처리 후, 남은 buffer 저장
        # 10000회 초과하면 중단 -> 현재까지 처리 안 된 것도 저장
        if gpt_call_count > gpt_call_limit:
            print(f"GPT 호출 횟수 10000회 초과. 중단합니다. 호출 횟수: {gpt_call_count}")
            # 남은 buffer 저장
            if len(episode_buffer) > 0:
                save_count += 1
                save_path = f"/home/zzzanghun/git/decision-transformer/gym/data/GPT_reward/gtp_reward_data_{save_count}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(episode_buffer, f)
                print(f"잔여 {len(episode_buffer)}개 에피소드 저장 완료 -> {save_path}")
            return


if __name__ == '__main__':
    get_traj()
