# uniform distribution
# 1000 packets
# window size - 14
# array of quantiles (left boundary, right boundary) (1,13) - >check algorithm of AIFO
# precomputed queues: generate queues for SP-AIFO and one queue for AIFO
# comparisons:
# 1.vs PIFO
# 2. check priority vs AIFO: priority range for example packets ranked 1 - 5 (changeable) -> count how many packets
# -> check how many of those packets are in the 100 first places (changeable)
# 3. check how many drops compared to AIFO

from algorithms import aifo, sp_aifo, aifo_with_fifo
from simulator import Simulator
from plotter import Plotter

import threading
import time
from threading import Thread
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Constants
N = 1000  # number of packets to generate
WIN_SIZE = 16  # window size of quantile
HIGH = 15
LOW = 1
SA_C = SA_NUM = 4  # capacity of each SP-AIFO queue and number of queues
K = (1 / 8)

LEFT = 0
RIGHT = 0.12


def create_queues_and_outputs():
    # SP-AIFO queues
    sa_queues = [deque() for _ in range(SA_NUM)]
    sa_outputs = [deque() for _ in range(SA_NUM)]
    # AIFO queue
    aifo_queue = deque()
    aifo_output = deque()
    # AIFO with FIFO queue
    aifo_with_fifo_queue = deque()
    aifo_with_fifo_output = deque()

    return sa_queues, sa_outputs, aifo_queue, aifo_output, aifo_with_fifo_queue, aifo_with_fifo_output


# uniformly assign queue bound
def precompute(sa_queues):
    queue_map = dict()
    i = 0
    j = 0
    while i < WIN_SIZE:
        queue_map[i + 1] = sa_queues[j]
        i += 1
        if i % (WIN_SIZE / SA_NUM) == 0:
            j += 1
    return queue_map


def create_simulation(simulator, aifo_queue, queue_map, aifo_with_fifo_queue,
                      keep_fifo, C, num_of_loops=10,
                      *apply):
    # C = 16  # capacity of AIFO queue for this simulation
    # simulator = Simulator(n=N, win_size=WIN_SIZE, high=HIGH, low=LOW,
    #                       right_bound=RIGHT)
    # sa_queues, sa_outputs, aifo_queue, aifo_output = create_queues_and_outputs()
    # queues = [aifo_queue, *sa_queues]
    # outputs = [aifo_output, *sa_outputs]
    # queue_map = precompute(sa_queues)

    for _ in range(num_of_loops):
        packets = simulator.generate_packets()
        # sa_queues, sa_outputs, aifo_queue, aifo_output = create_queues_and_outputs()
        # queues = [aifo_queue, *sa_queues]
        # outputs = [aifo_output, *sa_outputs]
        # queue_map = precompute(sa_queues)
        simulator.create_threads()
        sa_output = []
        # aifo_output, pifo, sa_output = simulate(C)
        aifo_pack = [aifo, [C, K, aifo_queue]]
        aifo_with_fifo_pack = [aifo_with_fifo,
                               [C, K, aifo_with_fifo_queue, keep_fifo]]
        sp_aifo_pack = [sp_aifo, [SA_C, K, queue_map, WIN_SIZE]]
        simulator.simulate(aifo_pack, sp_aifo_pack, aifo_with_fifo_pack)
        aifo_output = simulator.get_outputs()[0]
        aifo_with_fifo_output = simulator.get_outputs()[1]
        sa_outputs = simulator.get_outputs()[2:]
        for i in range(SA_NUM - 1, -1, -1):
            sa_output += list(sa_outputs[i])
            sa_outputs[i].clear()
        aifo_output_list = list(aifo_output)
        aifo_with_fifo_output_list = list(aifo_with_fifo_output)
        outputs = [aifo_output_list, aifo_with_fifo_output_list, sa_output]
        for func, args in apply:
            func(*(args + [outputs, packets]))

        aifo_output.clear()
        aifo_with_fifo_output.clear()
        keep_fifo.clear()


def apply_priority(X, Y_arr, outputs, packets):
    pifo = sorted(packets, reverse=True)
    outputs += [pifo]
    for i, x in enumerate(X):
        for Y, output in zip(Y_arr, outputs):
            Y[i].append(np.sum(output[::-1][:x]))




def apply_thrown(idx, aifo_thrown, aifo_with_fifo_thrown, keep_fifo, outputs,
                 packets):
    n = len(packets)
    # outputs = [aifo_output_list, aifo_with_fifo_output_list, sa_output]
    # for Y,output in zip(Y_arr,outputs[:2]):   #without sp-aifo cause it is const
    #     Y[idx].append(n - len(output))
    #     sa_thrown[idx].append(n - len(sa_output))
    # print("aifo output: " ,len(outputs[0]))
    # print("aifo fifo output: " ,len(outputs[1]))
    # print(len(keep_fifo))
    # print(keep_fifo)
    aifo_thrown[idx].append(n - len(outputs[0]))
    aifo_with_fifo_thrown[idx].append(n - len(list(keep_fifo) + outputs[1]))


def apply_thrown_v2(idx, aifo_thrown, aifo_with_fifo_thrown,sa_thrown, keep_fifo, outputs,
                 packets):
    n = len(packets)
    # outputs = [aifo_output_list, aifo_with_fifo_output_list, sa_output]
    # for Y,output in zip(Y_arr,outputs):   #without sp-aifo cause it is const
    #     Y[idx].append(n - len(output))
    #     sa_thrown[idx].append(n - len(sa_output))
    # print("aifo output: " ,len(outputs[0]))
    # print("aifo fifo output: " ,len(outputs[1]))
    # print(len(keep_fifo))
    # print(keep_fifo)
    aifo_thrown[idx].append(n - len(outputs[0]))
    aifo_with_fifo_thrown[idx].append(n - len(list(keep_fifo) + outputs[1]))
    sa_thrown[idx].append(n - len(outputs[2]))


def priority_simulation(plotter, simulator, queue_map, aifo_queue,
                        aifo_with_fifo_queue, keep_fifo):
    X = np.arange(100, 600, 100)
    x_len = len(X)
    aifo_Y = [[] for _ in range(x_len)]
    aifo_with_fifo_Y = [[] for _ in range(x_len)]
    sa_Y = [[] for _ in range(x_len)]
    pifo_Y = [[] for _ in range(x_len)]
    num_of_loops = 5  # todo:change

    C = 16  # capacity of AIFO queue for this simulation
    Y_arr = [aifo_Y, aifo_with_fifo_Y, sa_Y, pifo_Y]
    # simulation
    apply_pack = [apply_priority, [X, Y_arr]]
    create_simulation(simulator, aifo_queue, queue_map, aifo_with_fifo_queue,
                      keep_fifo, C, num_of_loops,
                      apply_pack)
    # create_simulation(X, aifo_Y, sa_Y, pifo_Y, num_of_loops)

    # calculate mean for each array size :100 -> 500
    for i in range(x_len):
        for Y in Y_arr:
            Y[i] = np.mean(Y[i])
        # sa_Y[i] = np.mean(sa_Y[i])
        # aifo_Y[i] = np.mean(aifo_Y[i])
        # pifo_Y[i] = np.mean(pifo_Y[i])

    # plot

    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO", "PIFO"]
    x_label = 'Amount of packets'
    y_label = 'Sum of rank'
    title = "Priority comparison - AIFO, AIFO with FIFO, SP-AIFO, PIFO"
    path = "C:\\Users\\tomer\\PycharmProject\\Archi\\results\\priority_compare.png"
    plotter.plot(X, Y_arr, labels, title, x_label, y_label, path)

def thrown_packets_simulation(simulator, plotter, queue_map, aifo_queue,
                              aifo_with_fifo_queue, keep_fifo):
    # Cs = np.arange(4, 34, 4)
    Cs = np.arange(16, 40, 4)
    # sa_thrown = [[] for _ in range(len(Cs))]
    aifo_thrown = [[] for _ in range(len(Cs))]
    aifo_with_fifo_thrown = [[] for _ in range(len(Cs))]
    sa_thrown = [49] * len(
        Cs)  # this number is the number of thrown packets with sp-pifo with the current settings
    Y_arr = [aifo_thrown, aifo_with_fifo_thrown, sa_thrown]

    num_of_loops = 5  # todo: change
    # apply_pack = [apply_thrown, [j, aifo_thrown, sa_thrown]]
    for j, C in enumerate(Cs):
        apply_pack = [apply_thrown,
                      [j, aifo_thrown, aifo_with_fifo_thrown, keep_fifo]]
        # for _ in range(5):
        create_simulation(simulator, aifo_queue, queue_map,
                          aifo_with_fifo_queue, keep_fifo, C, num_of_loops,
                          apply_pack)

        # sa_thrown[j].append(N - len(sa_output))
        # aifo_thrown[j].append(N - len(aifo_output))
        # sa_thrown[j] = np.mean(sa_thrown[j])
        aifo_thrown[j] = np.mean(aifo_thrown[j])
        aifo_with_fifo_thrown[j] = np.mean(aifo_with_fifo_thrown[j])

    # # plot_thrown_simulation(sa_thrown, aifo_thrown, Cs)
    # # plot
    # print("this is afet: " , len(keep_fifo))
    # print(sa_thrown)  # todo:erase
    # print(aifo_thrown)  # todo:erase
    # print(aifo_with_fifo_thrown)  # todo:erase

    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO"]
    x_label = 'AIFO capacity'
    y_label = 'Amount of packets'
    title = "Number of thrown packets - AIFO, AIFO with FIFO, SP-AIFO"
    path = "C:\\Users\\tomer\\PycharmProject\\Archi\\results\\thrown_compare.png"
    plotter.plot(Cs, Y_arr, labels, title, x_label, y_label, path)


def thrown_packet_simulation_with_time(simulator, plotter, queue_map,
                                       aifo_queue,
                                       aifo_with_fifo_queue, keep_fifo):
    X = np.arange(0.14, 0.24, 0.02)
    # X = np.arange(0.14, 0.18, 0.02)
    x_len = len(X)
    aifo_thrown = [[] for _ in range(x_len)]
    aifo_with_fifo_thrown = [[] for _ in range(x_len)]
    sa_thrown = [[] for _ in range(x_len)]

    C = 24  # AIFO and AIFO with FIFO capacity
    num_of_loops = 5  # todo:change

    Y_arr = [aifo_thrown, aifo_with_fifo_thrown, sa_thrown]
    # simulation
    for idx, x in enumerate(X):
        simulator.set_sleep_time(x)
        apply_pack = [apply_thrown_v2, [idx, aifo_thrown, aifo_with_fifo_thrown,sa_thrown, keep_fifo]]
        create_simulation(simulator, aifo_queue, queue_map, aifo_with_fifo_queue,
                      keep_fifo, C, num_of_loops, apply_pack)
        for Y in Y_arr:
            Y[idx] = np.mean(Y[idx])
    # for i in range(x_len):
    #     for Y in Y_arr:
    #         Y[i] = np.mean(Y[i])

    # plot
    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO"]
    x_label = 'AIFO capacity'
    y_label = 'Amount of packets'
    title = "Number of thrown packets - AIFO, AIFO with FIFO, SP-AIFO"
    path = "C:\\Users\\tomer\\PycharmProject\\Archi\\results\\thrown_compare_time.png"
    plotter.plot(X, Y_arr, labels, title, x_label, y_label, path)


def main():
    plotter = Plotter()
    simulator = Simulator(n=N, win_size=WIN_SIZE, high=HIGH, low=LOW,
                          right_bound=RIGHT)
    sa_queues, sa_outputs, aifo_queue, aifo_output, aifo_with_fifo_queue, aifo_with_fifo_output = create_queues_and_outputs()
    queues = [aifo_queue, aifo_with_fifo_queue, *sa_queues]
    outputs = [aifo_output, aifo_with_fifo_output, *sa_outputs]
    simulator.set_queues(queues)
    simulator.set_outputs(outputs)

    queue_map = precompute(sa_queues)
    keep_fifo = deque()

    # 1 check: priority
    # priority_simulation(plotter, simulator, queue_map, aifo_queue,
    #                     aifo_with_fifo_queue, keep_fifo)

    # 2 check: number of thrown packets as function of C

    thrown_packets_simulation(simulator, plotter, queue_map, aifo_queue,
                              aifo_with_fifo_queue, keep_fifo)

    # 3 better? number of thrown packets as function of insert/remove time
    thrown_packet_simulation_with_time(simulator, plotter, queue_map, aifo_queue,
                              aifo_with_fifo_queue, keep_fifo)

if __name__ == '__main__':
    main()
