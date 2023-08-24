import threading
import time
from threading import Thread
from typing import Any

import numpy as np
from collections import deque


class Simulator:

    def __init__(self, n, win_size, high, low, right_bound=None):
        """
        :param C:
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
        uniformly generate packets
        :return:
        """
        self.packets = np.random.uniform(self.low, self.high + 1,
                                         self.n).astype(int) / self.win_size
        return self.packets

    def set_queues(self, queues):
        self.queues = queues

    def set_outputs(self, outputs):
        self.outputs = outputs

    def get_queues(self):
        return self.queues

    def get_outputs(self):
        return self.outputs

    def set_sleep_time(self,right_bound):
        self.right_bound = right_bound

    def remove_to_output(self, output, arr):
        while not self.to_kill.is_set():
            tosleep = np.random.uniform(1, 5)
            time.sleep(tosleep / 25)
            if arr:
                output.appendleft(arr.pop())

        # pop everything the queue has before termination
        while arr:
            output.appendleft(arr.pop())
        print("Stopping as you wish.")

    def create_threads(self):
        n = len(self.queues)
        if n != len(self.outputs):
            print(
                "number of queues and number of outputs need to have same size.")
            return
        for i in range(n):
            thread = Thread(target=self.remove_to_output,
                            args=(self.outputs[i], self.queues[i]))
            self.threads.append(thread)

        return self.threads

    def simulate(self, *funcs_with_args):
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
