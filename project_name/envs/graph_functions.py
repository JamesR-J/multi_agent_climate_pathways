import numpy as np
import operator as op

import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from matplotlib.lines import Line2D

import functools as ft
import heapq as hq

from . import ays_model as ays

INFTY_SIGN = u"\u221E"

A_scale = 1
S_scale = 1e9
Y_scale = 1e12
tau_A = 50
tau_S = 50
beta = 0.03
beta_DG = 0.015
eps = 147
A_offset = 300
theta = 8.57e-5

rho = 2.
sigma = 4e12
sigma_ET = 2.83e12

phi = 4.7e10

current_state = [240, 7e13, 5e11]

color_list = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']

shelter_color = '#ffffb3'
management_options = ['default', 'DG', 'ET', 'DG+ET']
parameters = ['A', 'Y', 'S', 'Action', 'Reward']

reward_types = ['survive', 'survive_cost', 'desirable_region', 'rel_share', 'PB']
management_actions = [(False, False), (True, False), (False, True), (True, True)]
SMALLEST_SIZE = 10
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLEST_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLEST_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# current_state = [240, 7e13, 5e11]
current_state = [0, 7e13, 240]

# a small hack to make all the parameters available as global variables
ays.globalize_dictionary(ays.grid_parameters, module=ays)
ays.globalize_dictionary(ays.boundary_parameters, module=ays)


def create_figure_ays(top_down, label=None, colors=None, ax=None, ticks=True, plot_boundary=True,
                  reset=False, ax3d=None, fig3d=None):
    if not reset:
        if ax is None:
            fig3d = plt.figure(figsize=(12, 8))
            # ax3d = plt3d.Axes3D(fig3d)
            ax3d = fig3d.add_subplot(111, projection="3d")
        else:
            fig3d = None
            ax3d = ax

    if ticks == True:
        make_3d_ticks_ays(ax3d)
    else:
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

    A_PB = [10, 265]
    top_view = [25, 170]

    azimuth, elevation = 140, 15
    if top_down:
        azimuth, elevation = 180, 90

    ax3d.view_init(elevation, azimuth)

    S_scale = 1e9
    Y_scale = 1e12
    ax3d.set_xlabel("\n\nEmissions E  \n" + r"($GtC$)")
    ax3d.set_ylabel("\n\nEconomic output Y \n" + r"($1 \times 10^{12} USD/yr$)")
    if not top_down:
        ax3d.set_zlabel("\n\nExcess atmospheric carbon A \n" + r"($GtC$)")

    # Add boundaries to plot
    if plot_boundary:
        add_boundary(ax3d, sunny_boundaries=["planetary-boundary", "social-foundation"], model='ays', **ays.grid_parameters, **ays.boundary_parameters)

    ax3d.grid(False)

    legend_elements = []
    if label is None:
        # For Management Options
        for idx in range(len(management_options)):
            legend_elements.append(Line2D([0], [0], lw=2, color=color_list[idx], label=management_options[idx]))
    else:
        for i in range(len(label)):
            ax3d.scatter(*zip([0.5, 0.5, 0.5]), lw=1, color=colors[i], label=label[i])

    # For legend
    legend_elements.append(
        Line2D([0], [0], lw=2, label='current state', marker='o', color='w', markerfacecolor='red', markersize=15))
    # ax3d.legend(handles=legend_elements,prop={'size': 14}, bbox_to_anchor=(0.85,.90), fontsize=20,fancybox=True, shadow=True)

    return fig3d, ax3d


def add_boundary(ax3d, *, sunny_boundaries, add_outer=False, plot_boundaries=None, model='ays', **parameters):
    """show boundaries of desirable region"""

    if not sunny_boundaries:
        # nothing to do
        return

        # get the boundaries of the plot (and check whether it's an old one where "A" wasn't transformed yet
    if plot_boundaries is None:
        if "A_max" in parameters:
            a_min, a_max = 0, parameters["A_max"]
        elif "A_mid" in parameters:
            a_min, a_max = 0, 1
        w_min, w_max = 0, 1
        s_min, s_max = 0, 1
    else:
        a_min, a_max = plot_boundaries[0]
        w_min, w_max = plot_boundaries[1]
        s_min, s_max = plot_boundaries[2]

    if model == 'ricen':
        a_min = 0
        a_max = 20
        w_min = 0
        w_max = 750
        s_min = -10
        s_max = 15


    plot_pb = False
    plot_sf = False
    if "planetary-boundary" in sunny_boundaries:
        A_PB = parameters["A_PB"]
        if "A_max" in parameters:
            pass  # no transformation necessary
        elif "A_mid" in parameters:
            A_PB = A_PB / (A_PB + parameters["A_mid"])
        else:
            assert False, "couldn't identify how the A axis is scaled"
        if a_min < A_PB < a_max:
            plot_pb = True
    if "social-foundation" in sunny_boundaries:
        W_SF = parameters["W_SF"]
        W_SF = W_SF / (W_SF + parameters["W_mid"])
        if w_min < W_SF < w_max:
            plot_sf = True

    if model =='ricen':
        A_PB = 7

    if plot_pb and plot_sf:
        corner_points_list = [[[a_min, W_SF, A_PB],
                               [a_min, w_max, A_PB],
                               [a_max, w_max, A_PB],
                               [a_max, W_SF, A_PB],
                               ],
                              [[a_max, W_SF, s_min],
                               [a_min, W_SF, s_min],
                               [a_min, W_SF, A_PB],
                               [a_max, W_SF, A_PB],
                              ]]
    elif plot_pb:
        corner_points_list = [[[a_min, w_min, A_PB], [a_min, w_max, A_PB], [a_max, w_max, A_PB], [a_max, w_min, A_PB]]]
    elif plot_sf:
        corner_points_list = [[[a_min, W_SF, s_min], [a_max, W_SF, s_min], [a_max, W_SF, s_max], [a_min, W_SF, s_max]]]
    else:
        raise ValueError("something wrong with sunny_boundaries = {!r}".format(sunny_boundaries))

    boundary_surface_PB = plt3d.art3d.Poly3DCollection(corner_points_list, alpha=0.15)
    boundary_surface_PB.set_color("gray")
    boundary_surface_PB.set_edgecolor("gray")
    ax3d.add_collection3d(boundary_surface_PB)


def make_3d_ticks_ays(ax3d, boundaries=None, transformed_formatters=False, S_scale=1e9, Y_scale=1e12, num_a=12, num_y=12,
                  num_s=12, ):
    if boundaries is None:
        boundaries = [None] * 3

    transf = ft.partial(compactification, x_mid=current_state[0])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[0])

    # A- ticks
    if boundaries[0] is None:
        start, stop = 0, 20  # np.infty
        ax3d.set_xlim(0, 1)
    else:
        start, stop = inv_transf(boundaries[0])
        ax3d.set_xlim(*boundaries[0])

    ax3d.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3d.set_xticklabels([0, 4, 8, 12, 16, 20])

    # Y - ticks
    transf = ft.partial(compactification, x_mid=current_state[1])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[1])

    if boundaries[1] is None:
        start, stop = 0, np.infty
        ax3d.set_ylim(0, 1)
    else:
        start, stop = inv_transf(boundaries[1])
        ax3d.set_ylim(*boundaries[1])

    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, scale=Y_scale, start=start, stop=stop,
                                             num=num_y)
    if transformed_formatters:
        new_formatters = []
        for el, loc in zip(formatters, locators):
            if el:
                new_formatters.append("{:4.2f}".format(loc))
            else:
                new_formatters.append(el)
        formatters = new_formatters
    ax3d.yaxis.set_major_locator(ticker.FixedLocator(locators))
    ax3d.yaxis.set_major_formatter(ticker.FixedFormatter(formatters))

    transf = ft.partial(compactification, x_mid=current_state[2])
    inv_transf = ft.partial(inv_compactification, x_mid=current_state[2])

    # S ticks
    if boundaries[2] is None:
        start, stop = 0, np.infty
        ax3d.set_zlim(0, 1)
    else:
        start, stop = inv_transf(boundaries[2])
        ax3d.set_zlim(*boundaries[2])

    formatters, locators = transformed_space(transf, inv_transf, axis_use=True, start=start, stop=stop, num=num_s)
    if transformed_formatters:
        new_formatters = []
        for el, loc in zip(formatters, locators):
            if el:
                new_formatters.append("{:4.2f}".format(loc))
            else:
                new_formatters.append(el)
        formatters = new_formatters
    ax3d.zaxis.set_major_locator(ticker.FixedLocator(locators))
    ax3d.zaxis.set_major_formatter(ticker.FixedFormatter(formatters))


@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)


@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)


def transformed_space(transform, inv_transform, start=0, stop=np.infty, num=12, scale=1, num_minors=50, endpoint=True,
                      axis_use=False, boundaries=None, minors=False):
    add_infty = False
    if stop == np.infty and endpoint:
        add_infty = True
        endpoint = False
        num -= 1

    locators_start = transform(start)
    locators_stop = transform(stop)

    major_locators = np.linspace(locators_start,
                                 locators_stop,
                                 num,
                                 endpoint=endpoint)

    major_formatters = inv_transform(major_locators)
    # major_formatters = major_formatters / scale

    major_combined = list(zip(major_locators, major_formatters))
    # print(major_combined)

    if minors:
        _minor_formatters = np.linspace(major_formatters[0], major_formatters[-1], num_minors, endpoint=False)[1:]
        minor_locators = transform(_minor_formatters)
        minor_formatters = [np.nan] * len(minor_locators)
        minor_combined = list(zip(minor_locators, minor_formatters))
    # print(minor_combined)
    else:
        minor_combined = []
    combined = list(hq.merge(minor_combined, major_combined, key=op.itemgetter(0)))

    # print(combined)

    if not boundaries is None:
        combined = [(l, f) for l, f in combined if boundaries[0] <= l <= boundaries[1]]

    ret = tuple(map(np.array, zip(*combined)))
    if ret:
        locators, formatters = ret
    else:
        locators, formatters = np.empty((0,)), np.empty((0,))
    formatters = formatters / scale

    if add_infty:
        # assume locators_stop has the transformed value for infinity already
        locators = np.concatenate((locators, [locators_stop]))
        formatters = np.concatenate((formatters, [np.infty]))

    if not axis_use:
        return formatters

    else:
        string_formatters = np.zeros_like(formatters, dtype="|U10")
        mask_nan = np.isnan(formatters)
        if add_infty:
            string_formatters[-1] = INFTY_SIGN
            mask_nan[-1] = True
        string_formatters[~mask_nan] = np.round(formatters[~mask_nan], decimals=2).astype(int).astype("|U10")
        return string_formatters, locators



