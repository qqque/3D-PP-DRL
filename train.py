
import torch

import numpy as np
import time
import os

import logging
from pathlib import Path
from tensorboardX import SummaryWriter
import multiprocessing as mp
from multiprocessing import Queue
from arguments import Arguments
from agent import AgentPPO
from Agent_explore_env import Agent_explore_env

# =======================================================================================================================

def train_and_evaluate(args, action_queue_list, result_queue_list):
    args.init_before_training()

    '''init: Agent'''
    save_dir = args.save_dir
    writer = SummaryWriter("runs/" + save_dir)
    set_logging(save_dir)

    logging.info(f"{'Step':>10}{'maxG':>10} |"
                 f"{'expR':>10}{'expG':>10}{'objC':>10}{'objA':>10}{'logp':>10}")



    agent = AgentPPO(bin_size_x=args.bin_size_ds_x,
                     bin_size_y=args.bin_size_ds_y,
                     box_num=args.box_num,
                     plane_feature_num=args.plane_feature_num,
                     lr_actor=args.lr_actor,
                     lr_critic=args.lr_critic,
                     if_use_gae=args.if_per_or_gae,
                     load_model=args.load_model,
                     cwd=args.cwd)




    '''init Evaluator'''
    evaluator = Evaluator(args.cwd)

    '''init ReplayBuffer'''
    buffer = list()

    def update_buffer(s_a_n_r_m):
        buffer[:] = s_a_n_r_m  # (state, action, noise, reward, mask)
        _steps = s_a_n_r_m[3].shape[0]  # buffer[3] = r_sum
        _r_exp = s_a_n_r_m[3].mean()  # buffer[3] = r_sum
        return _steps, _r_exp

    '''start training'''
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    evaluator.total_step += args.load_step
    del args

    while evaluator.total_step < break_step:
        time_s1 = time.time()
        with torch.no_grad():
            trajectory_list = agent.explore_env_mp(action_queue_list, result_queue_list, target_step, reward_scale,
                                                   gamma)
            steps, r_exp = update_buffer(trajectory_list)
            evaluator.total_step += steps
            evaluator.save_model(agent.act,agent.cri,agent.goal_avg)

        print("The time of explore environment: %s" % (int(time.time() - time_s1)))
        time_s2 = time.time()

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)

        print("The time of update_net: %s" % (int(time.time() - time_s2)))

        evaluator.tensorboard_writer(r_exp,logging_tuple,agent.goal_avg,writer)


# =======================================================================================================================


class Evaluator:
    def __init__(self, cwd):
        self.g_max = -np.inf
        self.total_step = 0
        self.cwd = cwd

        logging.info(f"{'#' * 80}")
        logging.info(f"{'Step':>10}{'maxG':>10} |"
                     f"{'expR':>10}{'expG':>10}{'objC':>10}{'objA':>10}{'logp':>10}")

    def save_model(self, act, cri, g_exp):
        if g_exp > self.g_max:
            self.g_max = g_exp
            # save policy network in *.pth
            torch.save(act.state_dict(), f'{self.cwd}/actor.pth')
            torch.save(cri.state_dict(), f'{self.cwd}/critic.pth')
            # save policy and print
            logging.info(f"{self.total_step:10.4e}{self.g_max:10.4f} |")

    def tensorboard_writer(self,r_exp, log_tuple, g_exp, writer):
        writer.add_scalar("return_exp", g_exp, self.total_step)
        writer.add_scalar("critic_loss", log_tuple[0], self.total_step)
        writer.add_scalar("actor_loss", log_tuple[1], self.total_step)
        writer.add_scalar("log_prob", log_tuple[2], self.total_step)
        logging.info(f"{self.total_step:10.2e}{self.g_max:10.4f} |"
                     f"{r_exp:10.4f}{g_exp:10.4f}{''.join(f'{n:10.4f}' for n in log_tuple)}")
        with open(self.cwd + "last_step.txt", "w") as f:
            f.write("{}".format(self.total_step))



# =======================================================================================================================


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
        device = torch.device('cpu')
    return device


def set_logging(save_name):
    my_path = Path("./log")
    if not my_path.is_dir():
        os.makedirs(my_path)
    logging.basicConfig(filename=("./log/" + save_name + ".log"),
                        filemode="a",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s")


# =======================================================================================================================


if __name__ == '__main__':

    args = Arguments()
    args.device = set_device()

    action_queue_list = [Queue(maxsize=1) for _ in range(args.process_num)]
    result_queue_list = [Queue(maxsize=1) for _ in range(args.process_num)]
    process_list = list()

    for pi in range(args.process_num):
        t = mp.Process(target=Agent_explore_env,
                       args=(action_queue_list[pi],
                             result_queue_list[pi],
                             args.bin_size_x,
                             args.bin_size_y,
                             args.bin_size_z,
                             args.bin_size_ds_x,
                             args.bin_size_ds_y,
                             args.box_num,
                             args.min_factor,
                             args.max_factor,
                             args.plane_feature_num,
                             args.distance_threshold,
                             args.gap_filling,
                             ))

        process_list.append(t)
    [t.start() for t in process_list]

    train_and_evaluate(args, action_queue_list, result_queue_list)
