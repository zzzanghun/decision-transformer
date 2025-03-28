import tiktoken

# 2) cl100k_base (ChatGPT, GPT-4 등에서 주로 사용)
encoding = tiktoken.get_encoding("cl100k_base")

# 3) token 개수 세고 싶은 문자열(=prompt)을 text 변수에 넣는다.
text = """{
    "role": "You are the judge evaluating the drone's polynomial trajectory. Based on the step-by-step evaluation criteria below, produce a final score in the range of 0 to 1.",
    "instructions": [
      "1. The given data (grid map, the direction vector from the drone's current position to the target, the drone's current state, and the drone's predicted velocity up to 1.5 seconds) comes from the polynomial trajectory. Analyze this information comprehensively.",
      "2. Evaluate safety and efficiency independently, then calculate the overall score in the range of 0 to 1.",
      "3. Safety is evaluated based on the distance to obstacles, whether it passes through densely populated areas, and the possibility of collision.",
      "4. Efficiency is evaluated based on how much the trajectory deviates from the straight path to the target, the magnitude of velocity, and the alignment of the drone's predicted velocity with the target direction.",
      "5. Only consider the given trajectory information up to 1.5 seconds, but note that beyond this time, the drone may need to make a large detour or come close to obstacles, which introduces uncertainty.",
      "6. Therefore, from 1.5 seconds onward, use the predicted velocity to infer the drone's expected trajectory and evaluate whether it maintains a sufficient distance from obstacles.",
      "7. The current trajectory is generated every 0.5 seconds, and the drone's target is always outside the grid map. Thus, do not judge solely by the current data; you must also infer the trajectory beyond 1.5 seconds for evaluation.",
      "8. After balancing safety and efficiency, produce a final score in the range of 0 to 1.",
      "9. If there are no obstacles nearby, mainly evaluate the efficiency score based on the alignment between the target direction vector and the drone's predicted velocity vector. In this case, if the predicted velocity is 0.45 m/s or less, it is considered low efficiency.",
      "10. Output only a float value between 0 and 1.",
      "11. For reference, a very safe trajectory example scoring 0.1 and a very efficient trajectory example scoring 0.9 are provided. Keep in mind that if a trajectory has higher alignment with the target direction and predicted velocity but passes closer to obstacles, it can be considered more efficient compared to a very safe trajectory. Conversely, if it has lower alignment but maintains a greater distance from obstacles than a very efficient trajectory, it can be considered safer."
    ],
    "scoring": {
      "0": "A very safe but inefficient trajectory",
      "0.5": "A trajectory that balances safety and efficiency",
      "1": "A very efficient but less safe trajectory"
    },
    "input_data": {
      "grid_map": {
        "description": "100 × 100 grid map (10cm per cell). Rows represent the x-axis, columns represent the y-axis. As coordinates increase, the index decreases. For example, if the drone's position is (50, 50) and it moves +1m in both the x and y directions, it becomes (40, 40).",
        "size": "100x100",
        "resolution": "10cm per cell",
        "drone_position": [50, 50],
        "legend": {
          "0": "No obstacle",
          "1": "Obstacle",
          "2": "Predicted drone trajectory for 1.5 seconds"
        },
        "run_length_data_explanation": "Fill the indicated rows and cols range with the specified value. All unmentioned cells are 0.",
        "run_length_data": [
          { "rows": "20-36", "cols": "36-50", "value": "1" },
          { "rows": "20-36", "cols": "77-90", "value": "1" },
          { "rows": "23-36", "cols": "0-13", "value": "1" },
          { "rows": "59-76", "cols": "0-10", "value": "1" },
          { "rows": "59-74", "cols": "36-50", "value": "1" },
          { "rows": "59-75", "cols": "76-89", "value": "1" },
          { "rows": "98-98", "cols": "0-8", "value": "1" },
          { "rows": "50-50", "cols": "50-50", "value": "2" },
          { "rows": "50-50", "cols": "51-51", "value": "2" },
          { "rows": "50-50", "cols": "52-52", "value": "2" },
          { "rows": "50-50", "cols": "53-53", "value": "2" },
          { "rows": "50-50", "cols": "54-54", "value": "2" },
          { "rows": "50-50", "cols": "55-55", "value": "2" },
          { "rows": "50-50", "cols": "56-56", "value": "2" },
          { "rows": "50-50", "cols": "57-57", "value": "2" },
          { "rows": "50-50", "cols": "58-58", "value": "2" }
        ]
      },
      "drone_current_position_target_direction_and_state": {
        "description": "6D vector representing the drone's current state",
        "data": {
          "direction_to_target": "0.198,-0.98",
          "current_velocity": "0.346,-0.463",
          "current_acceleration": "0.027,0.014"
        }
      },
      "drone_predicted_velocity": {
        "description": "Drone's predicted velocity",
        "expected_velocity": {
          "description": "Predicted velocities from 0 to 1.5 seconds in 0.1-second intervals",
          "data": {
            "x_velocity": "[0.346,0.351,0.361,0.371,0.379,0.382,0.38,0.372,0.357,0.336,0.309,0.279,0.246,0.215,0.187,0.168]",
            "y_velocity": "[-0.463,-0.463,-0.467,-0.475,-0.488,-0.504,-0.522,-0.541,-0.56,-0.575,-0.585,-0.585,-0.574,-0.546,-0.496,-0.421]"
          }
        }
      }
    },
    "evaluation_criteria": {
      "priority": "The key factors determining the score are how much the future trajectory (marked as 2 on the grid map) deviates from the straight path toward the target, and how safely it avoids obstacles (marked as 1). Additionally, the alignment of the drone's predicted velocity with the target direction is important.",
      "safety": {
        "description": "Evaluate whether the drone's trajectory minimizes the risk of collision with obstacles.",
        "criteria": [
          "1. Average and minimum distances between the trajectory (2) and obstacles (1).",
          "2. In densely populated areas, even if the predicted velocity is not aligned with the target direction, if the drone detours around dense obstacles, consider high safety.",
          "3. The greater the distance maintained from obstacles, the higher the safety."
        ]
      },
      "efficiency": {
        "description": "Evaluate how quickly and directly the drone moves toward the target.",
        "criteria": [
          "1. Deviation from the straight path to the target: the closer to a direct path, the higher the efficiency.",
          "2. If obstacles are avoided minimally, efficiency is higher.",
          "3. When passing through dense areas, if the drone's velocity is well aligned with the target direction, consider it efficient.",
          "4. Consider the velocity, acceleration, and predicted velocity (whether it efficiently moves toward the target).",
          "5. The degree to which the predicted velocity is aligned with the target direction."
        ]
      }
    },
    "example_scenarios": {
      "very_safe_trajectory": {
        "description": "This path detours around obstacles greatly, making collision risk almost zero, but resulting in a lower alignment with the target direction and lower efficiency.",
        "grid_map": {
          "description": "100 × 100 grid map (10cm per cell). Rows represent the x-axis, columns represent the y-axis. As coordinates increase, the index decreases. For example, if the drone's position is (50, 50) and it moves +1m in both the x and y directions, it becomes (40, 40).",
          "size": "100x100",
          "resolution": "10cm per cell",
          "drone_position": [50, 50],
          "legend": {
            "0": "No obstacle",
            "1": "Obstacle",
            "2": "Drone's expected trajectory"
          },
          "run_length_data_explanation": "Fill the indicated rows and cols range with the specified value. All unmentioned cells are 0.",
          "run_length_data": "[{rows:15-29,cols:53-69,value:1},{rows:16-31,cols:91-98,value:1},{rows:18-32,cols:17-30,value:1},{rows:55-72,cols:16-30,value:1},{rows:56-69,cols:52-69,value:1},{rows:56-69,cols:91-98,value:1},{rows:95-98,cols:13-28,value:1},{rows:50-50,cols:50-50,value:2},{rows:51-51,cols:49-49,value:2},{rows:51-51,cols:50-50,value:2},{rows:52-52,cols:48-48,value:2},{rows:52-52,cols:49-49,value:2},{rows:53-53,cols:46-46,value:2},{rows:53-53,cols:47-47,value:2},{rows:53-53,cols:48-48,value:2},{rows:54-54,cols:44-44,value:2},{rows:54-54,cols:45-45,value:2},{rows:54-54,cols:46-46,value:2}]"
        },
        "drone_current_position_target_direction_and_state": {
          "description": "6D vector representing the drone's current state",
          "data": {
            "direction_to_target": "-0.999,0.033",
            "current_velocity": "-0.538,0.241",
            "current_acceleration": "-0.039,-0.074"
          }
        },
        "drone_predicted_velocity": {
          "description": "Drone's predicted velocity",
          "expected_velocity": {
            "description": "Predicted velocities from 0 to 1.5 seconds in 0.1-second intervals",
            "data": {
              "x_velocity": "[-0.538,-0.529,-0.501,-0.459,-0.409,-0.358,-0.308,-0.263,-0.225,-0.195,-0.173,-0.158,-0.149,-0.143,-0.136,-0.124]",
              "y_velocity": "[0.241,0.249,0.279,0.323,0.372,0.419,0.46,0.491,0.511,0.518,0.515,0.504,0.49,0.478,0.477,0.496]"
            }
          }
        },
        "evaluation": "This trajectory avoids obstacles by a large margin, resulting in almost no collision risk. However, it is less efficient due to a longer distance to the target.",
        "expected_score": 0.1
      },
      "very_efficient_trajectory": {
        "description": "This path avoids obstacles while maintaining a high alignment with the target direction. Current speed is somewhat low, but it increases over time, suggesting a quick arrival at the target. However, it passes relatively close to obstacles.",
        "grid_map": {
          "description": "100 × 100 grid map (10cm per cell). Rows represent the x-axis, columns represent the y-axis. As coordinates increase, the index decreases. For example, if the drone's position is (50, 50) and it moves +1m in both the x and y directions, it becomes (40, 40).",
          "size": "100x100",
          "resolution": "10cm per cell",
          "drone_position": [50, 50],
          "legend": {
            "0": "No obstacle",
            "1": "Obstacle",
            "2": "Drone's expected trajectory"
          },
          "run_length_data_explanation": "Fill the indicated rows and cols range with the specified value. All unmentioned cells are 0.",
          "run_length_data": "[{rows:13-35,cols:54-72,value:1},{rows:13-26,cols:90-98,value:1},{rows:21-34,cols:16-29,value:1},{rows:53-69,cols:49-70,value:1},{rows:54-74,cols:15-31,value:1},{rows:55-69,cols:88-98,value:1},{rows:95-98,cols:12-27,value:1},{rows:96-98,cols:52-64,value:1},{rows:50-50,cols:50-50,value:2},{rows:51-51,cols:49-49,value:2},{rows:51-51,cols:50-50,value:2},{rows:52-52,cols:49-49,value:2},{rows:53-53,cols:49-49,value:2},{rows:54-54,cols:49-49,value:2},{rows:55-55,cols:48-48,value:2},{rows:55-55,cols:49-49,value:2},{rows:56-56,cols:48-48,value:2},{rows:57-57,cols:48-48,value:2},{rows:58-58,cols:48-48,value:2}]"
        },
        "drone_current_position_target_direction_and_state": {
          "description": "Drone's current position and target direction vector, along with its state",
          "data": {
            "direction_to_target": "-0.544,0.839",
            "current_velocity": "-0.475,0.194",
            "current_acceleration": "0.007,0.011"
          }
        },
        "drone_predicted_velocity": {
          "description": "Drone's predicted velocity",
          "expected_velocity": {
            "description": "Predicted velocities from 0 to 1.5 seconds in 0.1-second intervals",
            "data": {
              "x_velocity": "[-0.475,-0.476,-0.479,-0.484,-0.49,-0.499,-0.508,-0.518,-0.529,-0.54,-0.552,-0.563,-0.574,-0.585,-0.595,-0.604]",
              "y_velocity": "[0.194,0.193,0.189,0.181,0.171,0.158,0.145,0.13,0.114,0.098,0.082,0.067,0.052,0.038,0.025,0.014]"
            }
          }
        },
        "evaluation": "This path avoids obstacles while aligning closely with the target direction. The current speed is low but increases over time, suggesting efficient travel to the target. However, it comes relatively close to obstacles, increasing collision risk.",
        "expected_score": 0.9
      }
    },
    "output_format": "Output only a float value between 0 and 1. Absolutely no other text, words, or messages—only the numeric value."
  }
  """

tokens = encoding.encode(text)
print("토큰 개수:", len(tokens))