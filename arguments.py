import numpy as np
import torch
import os


class Arguments:
    def __init__(self, ):
        self.cwd = None
        self.break_step = 2 ** 30

        self.load_model = True
        self.load_step = 0

        '''Arguments for training'''
        self.gamma = 0.99
        self.reward_scale = 2 ** 0
        self.lr_actor = 1e-5
        self.lr_critic = 1e-4
        self.soft_update_tau = 2 ** -8

        self.net_dim = 2 ** 9
        # num of transitions sampled from replay buffer.
        self.batch_size =1024
        self.repeat_times = 4  # collect target_step, then update network
        self.target_step = 1024 * 8  # repeatedly update network to keep critic's loss small
        self.max_memo = self.target_step  # capacity of replay buffer
        # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        self.if_per_or_gae = True
        self.num_threads = 2

        '''Arguments for evaluate'''
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        '''Arguments for Network'''
        self.d_model = 128
        self.n_head = 4
        self.d_inner = 1024
        self.nlayers = 3

        '''Arguments for Environment'''
        self.bin_size_x = 100
        self.bin_size_y = 100
        self.bin_size_z = 100000
        self.bin_size_ds_x = 10
        self.bin_size_ds_y = 10
        self.box_num = 50
        self.min_factor = 0.1
        self.max_factor = 0.5
        self.plane_feature_num = 6
        self.distance_threshold = 0
        self.gap_filling = False


        self.save_dir = "{:d}_{:d}_{:d}_{:d}_{:d}_{:d}".format(self.bin_size_x,self.bin_size_y,self.bin_size_z,self.box_num,
                                    int(self.min_factor*10),int(self.max_factor*10))

        '''Other Arguments'''
        # self.explore_num = 1
        # self.process_num = self.target_step // self.box_num // self.explore_num
        self.process_num = 16

    def init_before_training(self):

        if not os.path.exists("save"):
            os.makedirs("save")

        save_index = 0
        while True:
            save_dir = self.save_dir + "_{}".format(save_index)

            if not os.path.exists("save/"+save_dir):
                if save_index == 0:
                    self.load_model = False
                    print("No save file, The load_model is set to False")
                if self.load_model:
                    save_dir = self.save_dir + "_{}".format(save_index-1)
                else:
                    os.makedirs("save/"+save_dir)
                self.save_dir = save_dir
                self.cwd = "./" + "save/" + save_dir + "/"
                if self.load_model:
                    with open(self.cwd+"last_step.txt","r") as f:
                        self.load_step = int(f.read())
                        print("load step:",self.load_step)
                break
            else:
                save_index += 1

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

