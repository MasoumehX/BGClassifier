import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.integrate import odeint
from scipy.misc import derivative

PATH = "/home/masoumeh/Desktop/MasterThesis/Code/BodyGesturePatternDetection/docs/plots/"


def plot_scatter(data, x, y, labels, title, filename, directory):
    discrete_13_color_list = ["#9076ff",
                               "#f87100",
                               "#3793ff",
                               "#f10027",
                               "#01c689",
                               "#ff62b3",
                               "#00611d",
                               "#ba0052",
                               "#4edcbf",
                               "#72005a",
                               "#becd8a",
                               "#002d52",
                               "#f9ba6a"]

    fig = px.scatter(data, x=data[x], y=data[y], color=data[labels], color_discrete_sequence=discrete_13_color_list,
                     hover_data=[labels], title=title)

    path = os.path.join(PATH, directory)
    if not os.path.exists(path):
        os.mkdir(path)
        fig.write_html(path + "/" + filename + ".html")
        print(path + "/" + filename + ".html")
    else:
        fig.write_html(path + "/" + filename + ".html")
        print(path + "/" + filename + ".html")


def plot_line(x, y, filename, title, save, show):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.-')
    plt.ylabel("Y")
    plt.xlabel("X")
    ax.set_title(title)
    if save:
        plt.savefig(os.path.join(PATH, filename, ".png"))
    if show:
        plt.show()
    plt.close()


def plot_freq_power(first_freq, second_freq, title, filename, show=True, save=False):
    """A helper function to plot the frequency power (Zipf law)"""
    fig, ax = plt.subplots()
    plt.plot(first_freq, second_freq, marker='*', fillstyle='none', linestyle='none', color='m')
    plt.ylabel("Frequency")
    plt.xlabel("Tokens")
    plt.xticks(rotation=35)
    ax.set_title(title)
    if save:
        plt.savefig(os.path.join(PATH,filename,".png"))
    if show:
        plt.show()
    plt.close()



def system(vect, t):
    x, y = vect
    return [x - y - x * (x ** 2 + 5 * y ** 2), x + y - y * (x ** 2 + y ** 2)]


if __name__ == '__main__':

    vect0 = [(-2 + 4 * np.random.random(), -2 + 4 * np.random.random()) for i in range(5)]
    t = np.linspace(0, 100, 1000)

    color = ['red', 'green', 'blue', 'yellow', 'magenta']

    plot = plt.figure()

    for i, v in enumerate(vect0):
        sol = odeint(system, v, t)
        plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0] - sol[:-1, 0], sol[1:, 1] - sol[:-1, 1], scale_units='xy',
                   angles='xy', scale=1, color=color[i])

    plt.show()




