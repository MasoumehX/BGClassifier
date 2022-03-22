import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.express as px


def plot_scatter(data, x, y, labels, title, path, filename, directory):
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

    new_path = os.path.join(path, directory)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        print(new_path)
        fig.write_html(new_path + "/" + filename + ".html")
        print(new_path + "/" + filename + ".html")
    else:
        fig.write_html(new_path + "/" + filename + ".html")
        print(new_path + "/" + filename + ".html")


def plot_line(x, y, filename, title, path, save, show):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.-')
    plt.ylabel("Y")
    plt.xlabel("X")
    ax.set_title(title)
    if save:
        plt.savefig(os.path.join(path, filename, ".png"))
    if show:
        plt.show()
    plt.close()


def plot_power_law(first_freq, second_freq, title, path, filename, show=True, save=False):
    """A helper function to plot the frequency power (Zipf law)"""
    fig, ax = plt.subplots()
    plt.loglog(first_freq, second_freq, marker='*', fillstyle='none', linestyle='none', color='m')
    plt.xlabel("Frequency of words")
    plt.ylabel("frequency of frequency")
    ax.set_title(title)
    if save:
        plt.savefig(path+filename+".png")
    if show:
        plt.show()
    plt.close()


def plot_all_points_for_words(df, path):
    "A helper fucntion to plot the trajectories points for all words"

    # removing the points corresponding to the lower body
    non_related_points = [8, 9, 12, 10, 13, 11, 24, 23, 22, 21, 14, 19, 20]
    df = df[~df.point.isin(non_related_points)]
    df["point"] = df["point"].astype(str)

    # for all the words in the data
    word_groups = df.groupby(by="words")
    for wname, wgroup in word_groups:
        fname = 0
        groups = wgroup.groupby(by="name")
        for name, group in groups:
            plot_scatter(group, x="nx", y="ny", labels="point", path=path, filename=name + "_" + str(fname),
                         directory=wname, title="Scatter plot of points for word " + wname)
            fname += 1


def plot_model_loss(history,
                  title='Loss',
                  save=True,
                  show=False,
                  path=None,
                  filename=None):

    # Plot training & validation accuracy values
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    plt.title(title + "-" + filename)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig(path+filename+".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True,
                          save=True,
                          show=False,
                          path=None,
                          filename=None):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(filename)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=50)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save:
        plt.savefig(path+filename+".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

