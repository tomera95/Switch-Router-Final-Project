import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    This class plots the simulation results.

    """

    def __init__(self, fig_size=(12, 8), edge_color='grey',
                 title_weight='Bold',
                 labels_weight='Bold', title_font=18, label_font=15):
        """
        Plotter initializer
        :param fig_size: size of the figure to plot
        :param edge_color: color of edges between bars
        :param title_weight: title of plot weight
        :param labels_weight: labels of legend weight
        :param title_font: font of the title
        :param label_font: font of the x/y lables
        """
        self.label_font = label_font
        self.labels_weight = labels_weight
        self.title_weight = title_weight
        self.title_font = title_font
        self.fig_size = fig_size
        self.edge_color = edge_color
        self.bar_width = 0.1

    def plot(self, X, Y_arr, labels, title, x_label, y_label, path=None):
        """
        Plots the results using matplotlibs bar plot.
        :param X: x-axis of the plot
        :param Y_arr: array of y-axis
        :param labels: names of bars for the legend
        :param title: plots title
        :param x_label: x-axis label
        :param y_label: y-axis label
        :param path: path to save figure
        """
        fig = plt.subplots(figsize=self.fig_size)
        x_length = len(X)
        y_arr_length = len(Y_arr)

        # Set position of bar on X axis
        curr_br = np.arange(x_length)
        brs = [curr_br]
        for _ in range(1, y_arr_length):
            br = [x + self.bar_width for x in curr_br]
            brs.append(br)
            curr_br = br

        # Make the plot
        for i in range(y_arr_length):
            plt.bar(brs[i], Y_arr[i], width=self.bar_width,
                    edgecolor=self.edge_color, label=labels[i])

        # Adding Xticks
        plt.title(title, fontweight='bold', fontsize=18)
        plt.xlabel(x_label, fontweight='bold', fontsize=15)
        plt.ylabel(y_label, fontweight='bold', fontsize=15)
        plt.xticks([r + self.bar_width for r in range(x_length)],
                   [str(x) for x in X])

        plt.legend()
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()
