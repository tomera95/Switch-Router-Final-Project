"""
In this script we call run our simulations:
1. The first simulation compares all 4 algorithms: AIFO, AIFO with FIFO ,SP-AIFO
and PIFO. In this simulation we sum the rank of packets advanced asa function of number of packets.
2. The second simulation compares "dropping" algorithms: AIFO, AIFO with FIFO and SP-AIFO.
In this simulation we check how many packets were dropped as function of the capacity of AIFO and AIFO with FIFO.
3. The final simulation is similar to the second. the difference is that we change the time of arrival of packets
instead of the capacity.
"""

from algorithms import aifo, sp_aifo, aifo_with_fifo
from simulator import Simulator
from plotter import Plotter
import numpy as np
from collections import deque

# Constants
N = 1000  # number of packets to generate
WIN_SIZE = 16  # window size of quantile
HIGH = 15
LOW = 1
SA_C = SA_NUM = 4  # capacity of each SP-AIFO queue and number of queues
K = (1 / 8)  # k argument of all algorithms
LEFT = 0
RIGHT = 0.12


def create_queues_and_outputs():
    """
    Create queues and output queues for the algorithms implementation and simulation results.
    :return: queues and outputs
    """
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


def precompute(sa_queues):
    """
    Maps SP-AIFO queues to specific quantile rank range
    :param sa_queues: SP-AIFO queues.
    :return: queue map
    """
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
                      keep_fifo, C, num_of_loops,
                      *apply):
    """
    fixing inputs for Simulator to run.
    Uses apply functions of different simulations on the results
    :param simulator: simulator instance
    :param aifo_queue: AIFO queue
    :param queue_map: SP-AIFO queue map
    :param aifo_with_fifo_queue: AIFO with FIFO algorithm queue
    :param keep_fifo: FIFO queue
    :param C: AIFO and AIFO with FIFO capacity.
    :param num_of_loops: number of iterations to run simulation.
    :param apply: apply functions for the results of the simulation
    :return:
    """

    for _ in range(num_of_loops):
        # Get simulation ready
        packets = simulator.generate_packets()
        simulator.create_threads()
        sa_output = []
        aifo_pack = [aifo, [C, K, aifo_queue]]
        aifo_with_fifo_pack = [aifo_with_fifo,
                               [C, K, aifo_with_fifo_queue, keep_fifo]]
        sp_aifo_pack = [sp_aifo, [SA_C, K, queue_map, WIN_SIZE]]

        # Simulate
        simulator.simulate(aifo_pack, sp_aifo_pack, aifo_with_fifo_pack)

        # Rearrange results for applying function
        aifo_output = simulator.get_outputs()[0]
        aifo_with_fifo_output = simulator.get_outputs()[1]
        sa_outputs = simulator.get_outputs()[2:]
        for i in range(SA_NUM - 1, -1, -1):
            sa_output += list(sa_outputs[i])
            sa_outputs[i].clear()
        aifo_output_list = list(aifo_output)
        aifo_with_fifo_output_list = list(aifo_with_fifo_output)
        outputs = [aifo_output_list, aifo_with_fifo_output_list, sa_output]

        # Apply function on results
        for func, args in apply:
            func(*(args + [outputs, packets]))

        # Clear data for next iteration
        aifo_output.clear()
        aifo_with_fifo_output.clear()
        keep_fifo.clear()


def apply_priority(X, Y_arr, outputs, packets):
    """
    First simulation apply function. computes PIFO in the beginning, then summing
    ranks till x packets.
    :param X: array of number of packets
    :param Y_arr: array to keep each algorithm result
    :param outputs: results as lists
    :param packets: packets
    :return:
    """
    pifo = sorted(packets, reverse=True)
    outputs += [pifo]
    for i, x in enumerate(X):
        for Y, output in zip(Y_arr, outputs):
            Y[i].append(np.sum(output[::-1][:x]))


def apply_thrown(idx, aifo_thrown, aifo_with_fifo_thrown, sa_thrown, keep_fifo,
                 version, outputs,packets):
    """
    Second and third simulation apply function (for second simulation version is ,
    for third simulation version is 2)

    :param idx: iteration number
    :param aifo_thrown: array to keep number of thrown packets for AIFO
    :param aifo_with_fifo_thrown: array to keep number of thrown packets for AIFO with FIFO
    :param sa_thrown: array to keep number of thrown packets for SP-AIFO
    :param keep_fifo: FIFO queue of AIFO with FIFO
    :param version: distinguish between second and third simulation
    :param outputs: results as lists
    :param packets: packets
    :return:
    """
    n = len(packets)
    aifo_thrown[idx].append(n - len(outputs[0]))
    aifo_with_fifo_thrown[idx].append(n - len(list(keep_fifo) + outputs[1]))
    if version == 2:
        sa_thrown[idx].append(n - len(outputs[2]))


def priority_simulation(plotter, simulator, queue_map, aifo_queue,
                        aifo_with_fifo_queue, keep_fifo):
    """
    main function of the first simulation.
    Initializes variables for simulation, simulates and plots results.
    :param plotter: Plotter instance to plot results
    :param simulator: Simulator instance to simulate
    :param queue_map: SP-AIFO queue map
    :param aifo_queue: AIFO queue
    :param aifo_with_fifo_queue: AIFO with FIFO algorithm queue
    :param keep_fifo: FIFO queue of AIFO with FIFO
    :return:
    """
    X = np.arange(100, 600, 100)
    x_len = len(X)
    aifo_Y = [[] for _ in range(x_len)]
    aifo_with_fifo_Y = [[] for _ in range(x_len)]
    sa_Y = [[] for _ in range(x_len)]
    pifo_Y = [[] for _ in range(x_len)]
    num_of_loops = 10

    C = 16  # capacity of AIFO queue for this simulation
    Y_arr = [aifo_Y, aifo_with_fifo_Y, sa_Y, pifo_Y]
    # simulation
    apply_pack = [apply_priority, [X, Y_arr]]
    create_simulation(simulator, aifo_queue, queue_map, aifo_with_fifo_queue,
                      keep_fifo, C, num_of_loops,
                      apply_pack)

    # calculate mean for each array size: 100 -> 500
    for i in range(x_len):
        for Y in Y_arr:
            Y[i] = np.mean(Y[i])

    # plot
    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO", "PIFO"]
    x_label = 'Amount of packets'
    y_label = 'Sum of rank'
    title = "Priority comparison - AIFO, AIFO with FIFO, SP-AIFO, PIFO"
    path = " "   # insert your path here
    plotter.plot(X, Y_arr, labels, title, x_label, y_label, path)


def thrown_packets_simulation(simulator, plotter, queue_map, aifo_queue,
                              aifo_with_fifo_queue, keep_fifo):
    """
    main function of the second simulation.
    Initializes variables for simulation, simulates and plots results.
    :param plotter: Plotter instance to plot results
    :param simulator: Simulator instance to simulate
    :param queue_map: SP-AIFO queue map
    :param aifo_queue: AIFO queue
    :param aifo_with_fifo_queue: AIFO with FIFO algorithm queue
    :param keep_fifo: FIFO queue of AIFO with FIFO
    :return:
    """
    Cs = np.arange(16, 40, 4)
    aifo_thrown = [[] for _ in range(len(Cs))]
    aifo_with_fifo_thrown = [[] for _ in range(len(Cs))]
    sa_thrown = [49] * len(Cs)  # this number is the number of thrown packets with sp-pifo with the current settings.
    Y_arr = [aifo_thrown, aifo_with_fifo_thrown, sa_thrown]
    num_of_loops = 10

    for j, C in enumerate(Cs):
        apply_pack = [apply_thrown,
                      [j, aifo_thrown, aifo_with_fifo_thrown, None, keep_fifo,
                       1]]
        create_simulation(simulator, aifo_queue, queue_map,
                          aifo_with_fifo_queue, keep_fifo, C, num_of_loops,
                          apply_pack)
        # Calculate mean for each capacity
        aifo_thrown[j] = np.mean(aifo_thrown[j])
        aifo_with_fifo_thrown[j] = np.mean(aifo_with_fifo_thrown[j])

    # plot
    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO"]
    x_label = 'AIFO capacity'
    y_label = 'Amount of packets'
    title = "Number of thrown packets - AIFO, AIFO with FIFO, SP-AIFO"
    path = " "  # insert your path here
    plotter.plot(Cs, Y_arr, labels, title, x_label, y_label, path)


def thrown_packet_simulation_with_time(simulator, plotter, queue_map,
                                       aifo_queue,
                                       aifo_with_fifo_queue, keep_fifo):
    """
    main function of the third simulation.
    Initializes variables for simulation, simulates and plots results.
    :param plotter: Plotter instance to plot results
    :param simulator: Simulator instance to simulate
    :param queue_map: SP-AIFO queue map
    :param aifo_queue: AIFO queue
    :param aifo_with_fifo_queue: AIFO with FIFO algorithm queue
    :param keep_fifo: FIFO queue of AIFO with FIFO
    :return:
    """
    times = np.arange(0.14, 0.24, 0.02)
    times_len = len(times)
    aifo_thrown = [[] for _ in range(times_len)]
    aifo_with_fifo_thrown = [[] for _ in range(times_len)]
    sa_thrown = [[] for _ in range(times_len)]

    C = 24  # AIFO and AIFO with FIFO capacity for this simulation
    num_of_loops = 10

    Y_arr = [aifo_thrown, aifo_with_fifo_thrown, sa_thrown]
    # simulation
    for j, time in enumerate(times):
        simulator.set_sleep_time(time)
        apply_pack = [apply_thrown,
                      [j, aifo_thrown, aifo_with_fifo_thrown, sa_thrown,
                       keep_fifo, 2]]
        create_simulation(simulator, aifo_queue, queue_map,
                          aifo_with_fifo_queue,
                          keep_fifo, C, num_of_loops, apply_pack)
        # calculate mean for each time
        for Y in Y_arr:
            Y[j] = np.mean(Y[j])

    # plot
    labels = ["AIFO", "AIFO with FIFO", "SP-AIFO"]
    x_label = 'AIFO capacity'
    y_label = 'Amount of packets'
    title = "Number of thrown packets - AIFO, AIFO with FIFO, SP-AIFO"
    path = " "  # insert your path here
    plotter.plot(times, Y_arr, labels, title, x_label, y_label, path)


def main():
    """
    main function - Initializes Plotter and Simulator instances and calls 3 simulation functions.
    :return:
    """
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
    priority_simulation(plotter, simulator, queue_map, aifo_queue,
                        aifo_with_fifo_queue, keep_fifo)

    # 2 check: number of thrown packets as function of C

    thrown_packets_simulation(simulator, plotter, queue_map, aifo_queue,
                              aifo_with_fifo_queue, keep_fifo)

    # 3 better? number of thrown packets as function of insert/remove time
    thrown_packet_simulation_with_time(simulator, plotter, queue_map,
                                       aifo_queue,
                                       aifo_with_fifo_queue, keep_fifo)


if __name__ == '__main__':
    main()
