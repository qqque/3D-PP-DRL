import numpy as np

class Env():
    def __init__(self, bin_size_x, bin_size_y, bin_size_z, bin_size_ds_x, bin_size_ds_y, box_num, min_factor,
                 max_factor, feature_num=6, distance_threshold=0, gap_filling=False):

        self.max_step = 8192

        # Environment Parameters
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.bin_size_z = bin_size_z
        self.bin_size_ds_x = bin_size_ds_x
        self.bin_size_ds_y = bin_size_ds_y
        self.box_num = box_num
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.feature_num = feature_num + 1
        self.distance_threshold = distance_threshold
        self.gap_filling = gap_filling
        self.offline = True


        # Environment Variables
        self.gap = 0
        self.total_volume = 0
        self.max_index = None
        self.packing_result = []
        self.residual_box_num = box_num
        self.total_trunc_height = 0


        # Environment Constants
        self.rotation_matrix = np.array([[[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]],
                                         [[1, 0, 0],
                                          [0, 0, 1],
                                          [0, 1, 0]],
                                         [[0, 1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]],
                                         [[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]],
                                         [[0, 1, 0],
                                          [0, 0, 1],
                                          [1, 0, 0]],
                                         [[0, 0, 1],
                                          [0, 1, 0],
                                          [1, 0, 0]]])

        self.rotation_matrix_all = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                                             [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                                             [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]])

        self.original_bin_height = np.zeros((self.bin_size_x, self.bin_size_y), dtype=np.float32)
        self.original_state_bin = self.get_bin_feature(self.original_bin_height)
        self.block_size_x = self.bin_size_x // self.bin_size_ds_x
        self.block_size_y = self.bin_size_y // self.bin_size_ds_y

        if self.bin_size_ds_y < self.bin_size_y or self.bin_size_ds_x < self.bin_size_x:
            self.original_state_bin_ds, self.original_max_index = self.down_sampling(self.original_state_bin)

    def get_distance(self, height_feature, x_ori, is_equal=True, neg=False, threshold=0):

        plane_feature = np.ones_like(height_feature)
        if x_ori:
            count = np.ones((1, self.bin_size_y))
            if neg:
                for x_index in range(1, self.bin_size_x):
                    comparison = abs(height_feature[x_index, :] - height_feature[x_index - 1, :]) <= threshold
                    count = np.where(comparison, count + 1, 0)
                    plane_feature[x_index, :] = count * 1
            else:
                for x_index in range(self.bin_size_x - 2, -1, -1):
                    if is_equal:
                        comparison = abs(height_feature[x_index, :] - height_feature[x_index + 1, :]) <= threshold
                    else:
                        comparison = height_feature[x_index, :] >= height_feature[x_index + 1, :]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[x_index, :] = count * 1
        else:
            count = np.ones((1, self.bin_size_x))
            if neg:
                for y_index in range(1, self.bin_size_y):
                    comparison = abs(height_feature[:, y_index] - height_feature[:, y_index - 1]) <= threshold
                    count = np.where(comparison, count + 1, 0)
                    plane_feature[:, y_index] = count * 1
            else:
                for y_index in range(self.bin_size_y - 2, -1, -1):
                    if is_equal:
                        comparison = abs(height_feature[:, y_index] - height_feature[:, y_index + 1]) <= threshold
                    else:
                        comparison = height_feature[:, y_index] >= height_feature[:, y_index + 1]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[:, y_index] = count * 1

        return plane_feature

    def get_bin_feature(self, height_feature):
        h_feature = height_feature
        p_feature_x = self.get_distance(h_feature, x_ori=True)
        p_feature_y = self.get_distance(h_feature, x_ori=False)
        p_feature_x_neg = self.get_distance(h_feature, x_ori=True, neg=True)
        p_feature_y_neg = self.get_distance(h_feature, x_ori=False, neg=True)
        p_feature_x_gap = self.get_distance(h_feature, x_ori=True, is_equal=False)
        p_feature_y_gap = self.get_distance(h_feature, x_ori=False, is_equal=False)

        if self.feature_num == 5:
            bin_feature_state = np.stack(
                [h_feature, p_feature_x, p_feature_y, p_feature_x_neg, p_feature_y_neg],
                -1)
        else:
            bin_feature_state = np.stack(
                [h_feature, p_feature_x, p_feature_y, p_feature_x_neg, p_feature_y_neg, p_feature_x_gap,
                 p_feature_y_gap],
                -1)

        return bin_feature_state

    @staticmethod
    def generate_box_array(bin_size_x, bin_size_y, box_num, min_factor, max_factor):

        box_array_x = np.random.randint(int(bin_size_x * min_factor), int(bin_size_x * max_factor + 1), [box_num,])
        box_array_y = np.random.randint(int(bin_size_y * min_factor), int(bin_size_y * max_factor + 1), [box_num,])
        box_array_z = np.random.randint(int(min(bin_size_x,bin_size_y) * min_factor),
                                        int(max(bin_size_x,bin_size_y) * max_factor + 1),
                                        [box_num,])
        box_array = np.stack([box_array_x,box_array_y,box_array_z],-1)

        return box_array

    def down_sampling(self, bin_state):
        bin_size_x, bin_size_y, feature_num = bin_state.shape
        bin_state_split = np.stack(np.split(bin_state, self.bin_size_ds_x, 0), 0)
        bin_state_split = np.stack(np.split(bin_state_split, self.bin_size_ds_y, 2), 1).reshape(
            self.bin_size_ds_x * self.bin_size_ds_y, -1, feature_num)
        max_target = bin_state_split[:, :, 1] * bin_state_split[:, :, 2]
        max_idx = max_target.argmax(-1).reshape(-1, 1, 1)

        bin_state_ds = np.take_along_axis(bin_state_split, max_idx, 1).reshape(self.bin_size_ds_x, self.bin_size_ds_y,
                                                                               -1)

        return bin_state_ds, max_idx

    def down_sampling_mask(self, mask, max_index):
        box_num, ori_num, bin_size_x, bin_size_y = mask.shape
        mask_split = np.stack(np.split(mask, self.bin_size_ds_x, 2), 2)
        mask_split = np.stack(np.split(mask_split, self.bin_size_ds_y, 4), 3).reshape(box_num, ori_num,
                                                                                   self.bin_size_ds_x * self.bin_size_ds_y,
                                                                                   -1)
        mask_ds = np.take_along_axis(mask_split, max_index.reshape(1, 1, -1, 1), -1).reshape(box_num, ori_num,
                                                                                             self.bin_size_ds_x,
                                                                                             self.bin_size_ds_y)

        return mask_ds

    def reset(self, box_array=None):
        # reset parameter
        self.gap = 0
        self.total_volume = 0
        self.total_trunc_height = 0
        self.packing_result = []
        self.residual_box_num = self.box_num
        # generate state_box
        if box_array is None:
            box_array = self.generate_box_array(self.bin_size_x, self.bin_size_y, self.box_num, self.min_factor,
                                                self.max_factor)
        else:
            box_array = box_array * 1
        self.box_array = box_array * 1

        state_box = self.box_array

        # generate state_bin
        self.bin_height = self.original_bin_height * 1
        state_bin = self.original_state_bin * 1

        if self.bin_size_ds_y < self.bin_size_y or self.bin_size_ds_x < self.bin_size_x:  # need downsampling

            state_bin = self.original_state_bin_ds * 1
            self.max_index = self.original_max_index * 1
            packing_mask = self.get_mask_no_constraint(state_box, self.max_index)

        else:
            packing_mask = self.get_mask_no_constraint(state_box)

        self.state = (state_bin, state_box, packing_mask)

        return self.state

    def get_mask_no_constraint(self, state_box, max_index=None):
        x_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_x, 0, -1).reshape(-1,1) * self.block_size_x
        y_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_y, 0, -1) * self.block_size_y
        position_residual_size = np.stack([x_residual_size, y_residual_size], 2)

        available_box_num = self.residual_box_num

        box_array = state_box[:available_box_num] * 1  # residual_box_num x 3

        box_rotation_array = np.matmul(box_array, self.rotation_matrix_all).reshape(-1, 6,
                                                                                    3)  # residual_box_num x 6 x 3
        box_rotation_array = box_rotation_array[:, :, :2]

        if max_index is not None:
            max_index_size = np.stack([max_index / (self.block_size_y), max_index % (self.block_size_y)], -1)
            position_residual_size = position_residual_size - max_index_size.reshape(self.bin_size_ds_x,
                                                                                     self.bin_size_ds_y,
                                                                                     2)  # 10 x 10 x 2

        packing_available = box_rotation_array.reshape(-1, 6, 1, 1, 2) <= position_residual_size.reshape(1, 1,
                                                                                                         self.bin_size_ds_x,
                                                                                                         self.bin_size_ds_y,
                                                                                                         2)
        packing_available = packing_available.all(-1)

        packing_available = np.pad(packing_available, ((0, self.box_num - available_box_num), (0, 0), (0, 0), (0, 0)),
                                   constant_values=False)

        return ~packing_available

    def step(self, action: tuple, debug=True):
        a_i, a_xy, a_r = action

        if self.max_index is not None:
            sub_index = int(self.max_index[a_xy])
            a_x = a_xy // self.bin_size_ds_y * self.block_size_x + sub_index // self.block_size_y
            a_y = a_xy % self.bin_size_ds_y * self.block_size_y + sub_index % self.block_size_y
        else:
            a_x = a_xy // self.bin_size_y
            a_y = a_xy % self.bin_size_y

        box_shape = self.box_array[a_i]
        box_rotation_shape = np.matmul(box_shape, self.rotation_matrix_all).reshape(6, 3)
        box_shape_rot = box_rotation_shape[a_r]
        box_length, box_width, box_height = map(int, box_shape_rot)
        if debug:
            assert a_x + box_length <= self.bin_size_x and a_y + box_width <= self.bin_size_y
        place_pack = self.bin_height[a_x:a_x + box_length, a_y:a_y + box_width]
        a_z = np.max(place_pack)
        place_pack[:, :] = float(np.max(place_pack) + box_height)
        if self.bin_size_z < 10000 and debug:
            assert np.max(place_pack) <= self.bin_size_z
        self.packing_result.append([box_length, box_width, box_height, a_x, a_y, a_z])

        # generate state_box
        self.box_array = np.delete(self.box_array, a_i, 0)
        self.box_array = np.pad(
            self.box_array, ((0, self.box_num - self.box_array.shape[0]), (0, 0)), constant_values=-1e9)
        state_box = self.box_array

        self.residual_box_num -= 1

        # generate state_bin


        bin_height = self.bin_height
        state_bin = self.get_bin_feature(bin_height)
        if self.bin_size_ds_y < self.bin_size_y or self.bin_size_ds_x < self.bin_size_x:
            state_bin_ds, self.max_index = self.down_sampling(state_bin)

        if self.bin_size_ds_y < self.bin_size_y or self.bin_size_ds_x < self.bin_size_x:  # need downsampling
            if self.residual_box_num != 0:
                state_bin, self.max_index = self.down_sampling(state_bin)
                packing_mask = self.get_mask_no_constraint(state_box,self.max_index)
            else:
                state_bin, self.max_index = self.down_sampling(state_bin)
                packing_mask = np.ones((self.box_num, 6, self.bin_size_ds_x, self.bin_size_ds_y))
        else:
            if self.residual_box_num != 0:
                packing_mask = self.get_mask_no_constraint(state_box)
            else:
                packing_mask = np.ones((self.box_num, 6, self.bin_size_ds_x, self.bin_size_ds_y))

        self.packing_mask = packing_mask
        self.state = (state_bin, state_box, packing_mask)

        # reward
        self.total_volume += box_length * box_width * box_height
        total_bin_volume = self.bin_size_x * \
                           self.bin_size_y * float(np.max(self.bin_height))

        new_gap = total_bin_volume - self.total_volume
        reward = (self.gap - new_gap) / self.bin_size_x / self.bin_size_y
        self.gap = new_gap
        if self.bin_size_z < 10000:
            total_bin_volume = self.bin_size_x * self.bin_size_y * self.bin_size_z
        self.use_ratio = self.total_volume / total_bin_volume * 100

        # done
        if self.residual_box_num == 0 or packing_mask.all():
            done = True
            if self.bin_size_z < 10000:
                reward = reward - (self.bin_size_z - np.max(self.bin_height))
        else:
            done = False
        return self.state, reward, done

