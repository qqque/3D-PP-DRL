import torch
import time
from network import ActorNetwork
from environment import Env
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from multiprocessing import Queue
import numpy as np
from Agent_explore_env import solve_problem
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plot import plotResult

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plotBox(x, y, z, dx, dy, dz, ax, color=None):
    verts = [(x, y, z), (x, y + dy, z), (x + dx, y + dy, z), (x + dx, y, z),
             (x, y, z + dz), (x, y + dy, z + dz), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
    poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
    x, y, z = zip(*verts)
    ax.add_collection3d(Poly3DCollection(
        poly3d, facecolors=color, linewidths=1, edgecolors='black'))


# ========================================================================================================


def outputResult(packingResult, bin_size=100):
    '''
    packingResult: [box1,box2,box3,...]
    box1:[l,w,h,p_x,p_y,p_z]
    '''
    # step1: check the correct of outputData: whether any box overlap with other box
    box_num = len(packingResult)
    for i in range(box_num - 1):
        for j in range(i + 1, box_num):
            box_i = np.array(packingResult[i])
            box_j = np.array(packingResult[j])
            box_i_c = box_i[0:3] / 2 + box_i[3:]
            box_j_c = box_j[0:3] / 2 + box_j[3:]
            is_overlap = (np.abs(box_j_c - box_i_c) < (box_i[0:3] + box_j[0:3]) / 2).all()
            if is_overlap:
                raise Exception("物品装载重叠，请检查结果")

    # step2: calculate the use rate
    packingArray = np.array(packingResult)
    boxHeightCo = packingArray[:, 2] + packingArray[:, 5]
    max_height = np.max(boxHeightCo)
    ur = packingArray[:, :3].prod(1).sum() / bin_size / bin_size / max_height
    print(ur)

    fig = plt.figure()

    ax = Axes3D(fig)
    ax.set_xlim(0, bin_size)
    ax.set_ylim(0, bin_size)
    ax.set_zlim(0, max_height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("uR:%s%%" % (ur.item() * 100), fontproperties="SimHei")
    for box in packingResult:
        color = (random.random(), random.random(), random.random(), 1)
        length, width, height, start_l, start_w, start_h = box
        plotBox(start_l, start_w, start_h,
                length, width, height, ax, color)

    plt.show()


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
        device = torch.device('cpu')
    return device


# =======================================================================================================================


if __name__ == '__main__':

    # parameter
    sample_num = 16
    test_num = 1024
    box_num = 100
    bin_size_x = 100
    bin_size_y = 100
    bin_size_z = 10000000
    bin_size_ds_x=10
    bin_size_ds_y=10
    min_factor = 0.1
    max_factor = 0.5

    distance_threshold = 0
    plane_feature_num = 6
    gap_filling = False
    load_file_path = "save/400_200_100000_50_1_5_0/actor.pth"
    device = set_device()

    # Create Environment
    action_queue_list = [Queue(maxsize=1) for _ in range(sample_num)]
    result_queue_list = [Queue(maxsize=1) for _ in range(sample_num)]


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
    gap_filling=gap_filling,)

    box_array_list = [Env.generate_box_array(bin_size_x, bin_size_y, box_num, min_factor, max_factor) for _ in
                      range(test_num)]

    process_list = list()
    for pi in range(sample_num):
        t = mp.Process(target=solve_problem, args=(action_queue_list[pi], result_queue_list[pi], box_array_list, env))
        process_list.append(t)
    [t.start() for t in process_list]

    act = ActorNetwork(bin_size_ds_x,bin_size_ds_y, box_num, plane_feature_num=plane_feature_num,binary_dim=10).to(device)

    act.load_state_dict(torch.load(load_file_path, map_location=device))

    [action_queue_list[pi].put(False) for pi in range(sample_num)]
    result_list = [result_queue.get() for result_queue in result_queue_list]
    state_list = [result[0] for result in result_list]

    total_time = 0
    ur_list = []
    packing_result_list = []

    for i in range(test_num):

        time_start = time.time()
        for j in range(box_num):
            state = list(map(list, zip(*state_list)))
            state = [torch.as_tensor(np.array(s), dtype=torch.float32, device=device) for s in state]
            actions, probs = act.get_action(state)
            action_list = np.array([act.detach().cpu().numpy() for act in actions]).transpose()
            action_int_list = action_list.tolist()
            [action_queue_list[pi].put(action_int_list[pi]) for pi in range(sample_num)]
            result_list = [result_queue.get() for result_queue in result_queue_list]
            result_list = list(map(list, zip(*result_list)))
            state_list = result_list[0]

            if result_list[2][0]:
                time1 = time.time() - time_start
                avg_ur = sum(result_list[3]) / sample_num
                ur_list.append(max(result_list[3]))
                packing_result_list.append(result_list[4][np.array(result_list[3]).argmax()])
                total_time += time1
                break

    use_ratio = sum(ur_list) / test_num
    use_time = total_time / test_num
    print("average use ratio:", use_ratio)
    print("average time:", use_time)

    [t.join() for t in process_list]

    bestResult = packing_result_list[np.array(ur_list).argmax()]
    worseResult = packing_result_list[np.array(ur_list).argmin()]
    plotResult(bestResult, bin_size_x,bin_size_y)
    print("The best use ratio of instances is:%.2f%%" % (max(ur_list)))
    plotResult(worseResult, bin_size_x,bin_size_y)
    print("The worse use ratio of instances is:%.2f%%" % (min(ur_list)))
