import threading
import time
from threading import Thread
import numpy as np


class Simulator:
    """
    This class implements simulations of packet scheduling algorithms
    with rank as quantile.
    """
    def __init__(self, n, win_size, high, low, right_bound=None):
        """
        Simulator initializer.
        :param n: number of packets
        :param win_size: window size
        :param high: max bound of ranks
        :param low: min bound of ranks
        :param right_bound: max bound of time to sleep (demonstrate packets arrival)
        """
        self.threads = []
        self.packets = None
        self.to_kill = threading.Event()
        self.left_bound = 0
        self.right_bound = right_bound
        self.low = low
        self.high = high
        self.win_size = win_size
        self.n = n

    def generate_packets(self):
        """
        Uniformly generate packets.
        :return: packets
        """
        self.packets = np.random.uniform(self.low, self.high + 1,
                                         self.n).astype(int) / self.win_size
        return self.packets

    def set_queues(self, queues):
        """
        Sets queues.
        :param queues: queues to set
        :return:
        """
        self.queues = queues

    def set_outputs(self, outputs):
        """
        Sets outputs.
        :param outputs: outputs to set
        :return:
        """
        self.outputs = outputs

    def get_queues(self):
        """
        :return: queues
        """
        return self.queues

    def get_outputs(self):
        """
        :return: outputs
        """
        return self.outputs

    def set_sleep_time(self, right_bound):
        """
        Sets the max time to sleep.
        :param right_bound: max time to sleep
        :return:
        """
        self.right_bound = right_bound

    def remove_to_output(self, output, queue):
        """
        Threads target function - clear queues randomly - moves to final output.
        :param output: final output of algorithm
        :param queue:
        :return:
        """
        while not self.to_kill.is_set():
            tosleep = np.random.uniform(1, 5)
            time.sleep(tosleep / 25)
            if queue:
                output.appendleft(queue.pop())

        # pop everything the queue has before termination
        while queue:
            output.appendleft(queue.pop())
        print("Stopping as you wish.")

    def create_threads(self):
        """
        Initializes the threads to run. number of threads depeneds on number of queues.
        :return:
        """
        n = len(self.queues)
        if n != len(self.outputs):
            print("Number of queues and number of outputs need to have same size.")
            return
        for i in range(n):
            thread = Thread(target=self.remove_to_output,
                            args=(self.outputs[i], self.queues[i]))
            self.threads.append(thread)

        return self.threads

    def simulate(self, *funcs_with_args):
        """
        Main function of this class. simulates packet scheduling algorithms.
        :param funcs_with_args:
        :return:
        """
        packets = self.packets
        if len(packets) == 0:
            print("You have to generate packets before calling this function")
            return

        # start threads
        for thread in self.threads:
            thread.start()

        # assigning packets to algorithms
        for packet in packets:
            if self.right_bound:
                time.sleep(
                    np.random.uniform(self.left_bound, self.right_bound))
            for func, kwargs in funcs_with_args:
                func(*([packet] + kwargs))

        # stop threads
        self.to_kill.set()
        for thread in self.threads:
            thread.join()

        self.threads.clear()
        self.to_kill.clear()
