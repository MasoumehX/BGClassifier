import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

PATH = "/home/masoumeh/Desktop/Thesis/BodyGesturePatternDetection/docs/plots/"


def plot_freq_power(first_freq, second_freq, show=True, save=False, title=""):
    """A helper function to plot the frequency power (Zipf law)"""
    fig, ax = plt.subplots()
    plt.plot(first_freq, second_freq, marker='*', fillstyle='none', linestyle='none', color='m')
    plt.ylabel("Frequency")
    plt.xlabel("Tokens")
    plt.xticks(rotation=35)
    ax.set_title("Frequency Distribution of (word) Types in the corpus")
    if save:
        plt.savefig(PATH+"plot_freq_power_data" + title + ".png")
    if show:
        plt.show()
    plt.close()


def plot_3d_trajectory(x, y, z):

    # References
    # https://www.bragitoff.com/2020/10/3d-trajectory-animated-using-matplotlib-python/

    # ANIMATION FUNCTION
    def func(num, dataSet, line, redDots):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        redDots.set_data(dataSet[0:2, :num])
        redDots.set_3d_properties(dataSet[2, :num])
        return line

    # THE DATA POINTS
    dataSet = np.array([x, y, z])
    numDataPoints = len(z)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = Axes3D(fig)
    redDots = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='r', marker='o')[0]  # For scatter plot
    # NOTE: Can't pass empty arrays into 3d version of plot()
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0]  # For line plot

    # AXES PROPERTIES]
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('time')
    ax.set_title('Trajectory of electron for E vector along [120]')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet, line, redDots), interval=50,
                                       blit=False)
    # line_ani.save(r'Animation.mp4')

    plt.show()