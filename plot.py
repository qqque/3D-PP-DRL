import plotly.graph_objects as go
import numpy as np
import random


def cube_trace(x, y, z, dx, dy, dz, scale, color=None):
    line = go.Scatter3d(
        x=[x, x + dx, x + dx, x, x, x, x + dx, x + dx, x, x, x + dx, x + dx, x + dx, x + dx, x, x],
        y=[y, y, y + dy, y + dy, y, y, y, y + dy, y + dy, y, y, y, y + dy, y + dy, y + dy, y + dy],
        z=[z, z, z, z, z, z + dz, z + dz, z + dz, z + dz, z + dz, z + dz, z, z, z + dz, z + dz, z],

        marker=dict(size=1),
        line=dict(color='black', width=3)
    )

    w = scale

    x += w
    y += w
    z += w
    dx -= 2 * w
    dy -= 2 * w
    dz -= 2 * w

    climit = 100

    colors = 'rgb(%s,%s,%s)' % (255 - color[0] * climit, 255 - color[1] * climit, 255 - color[2] * climit)
    surface = go.Mesh3d(
        # 8 vertices of a cube
        x=[x, x, x + dx, x + dx, x, x, x + dx, x + dx],
        y=[y, y + dy, y + dy, y, y, y + dy, y + dy, y],
        z=[z, z, z, z, z + dz, z + dz, z + dz, z + dz],

        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=1,
        color=colors,
        flatshading=True,
    )

    return surface, line


def plotResult(packingResult, bin_size_x=100,bin_size_y=100):
    '''
    packingResult: [box1,box2,box3,...]
    box1:[l,w,h,p_x,p_y,p_z]
    '''
    # step1: check the correct of outputData: whether any box overlap with other box.
    box_num = len(packingResult)
    for i in range(box_num - 1):
        for j in range(i + 1, box_num):
            box_i = np.array(packingResult[i])
            box_j = np.array(packingResult[j])
            box_i_c = box_i[0:3] / 2 + box_i[3:]
            box_j_c = box_j[0:3] / 2 + box_j[3:]
            is_overlap = (np.abs(box_j_c - box_i_c) < (box_i[0:3] + box_j[0:3]) / 2).all()
            if is_overlap:
                raise Exception("Error: Boxes overlap! Please check the result.")

    # step2: calculate the use rate.
    packingArray = np.array(packingResult)
    boxHeightCo = packingArray[:, 2] + packingArray[:, 5]
    max_height = np.max(boxHeightCo)
    ur = packingArray[:, :3].prod(1).sum() / bin_size_x / bin_size_y / max_height
    print("[The use ratio of result is: %.2f%%]"%(ur*100))

    # step3: plot the result.
    traces = []
    for box in packingResult:
        color = (random.random(), random.random(), random.random())
        length, width, height, start_l, start_w, start_h = box
        scale = 0.0007 * max(bin_size_y,bin_size_x)
        surface, line = cube_trace(start_l, start_w, start_h, length, width, height, scale, color)
        traces.append(surface)
        traces.append(line)

    _, border = cube_trace(0, 0, 0, bin_size_x, bin_size_y, max_height,scale, (0, 0, 0))

    traces.append(border)

    fig = go.Figure(data=traces)

    fig.update_layout(scene=dict(
        xaxis=dict(
            showbackground=False,
            color='white'
        ),
        yaxis=dict(
            showbackground=False,
            color='white'
        ),
        zaxis=dict(
            showbackground=False,
            color='white'
        ),
    ))

    fig.show()
