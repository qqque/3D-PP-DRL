import time

from environment import Env


def Agent_explore_env(action_queue,
                      result_queue,
                      bin_size_x,
                      bin_size_y,
                      bin_size_z,
                      bin_size_ds_x,
                      bin_size_ds_y,
                      box_num,
                      min_factor,
                      max_factor,
                      plane_feature_num,
                      distance_threshold=0, gap_filling=False, iter_num=10000000000):

    env = Env(bin_size_x = bin_size_x,
              bin_size_y = bin_size_y,
              bin_size_z = bin_size_z,
              bin_size_ds_x = bin_size_ds_x,
              bin_size_ds_y = bin_size_ds_y,
              box_num=box_num,
              min_factor=min_factor,
              max_factor=max_factor,
              feature_num=plane_feature_num,
              distance_threshold=distance_threshold,
              gap_filling=gap_filling)

    iter_num = iter_num * box_num

    for _ in range(iter_num):
        action = action_queue.get()
        if action is False:
            state = env.reset()
            result_queue.put((state, 0, 0))
            action = action_queue.get()
        next_state, reward, done = env.step(action)
        if done:
            use_ratio = env.use_ratio
            next_state = env.reset()
        else:
            use_ratio = 0
        result_queue.put((next_state, reward, done, use_ratio))


def solve_problem(action_queue, result_queue, box_array_list, env):
    for box_array in box_array_list:
        done = False
        while not done:
            action = action_queue.get()
            if action is False:
                state = env.reset(box_array)
                result_queue.put((state, 0, 0))
                action = action_queue.get()
            next_state, reward, done = env.step(action)
            if done:
                use_ratio = env.use_ratio
                packing_result = env.packing_result
                next_state = env.reset(box_array)
            else:
                use_ratio = 0
                packing_result = 0
            result_queue.put((next_state, reward, done, use_ratio, packing_result))