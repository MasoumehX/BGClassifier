import os
import matplotlib.pyplot as plt
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


def plot_freq_power(first_freq, second_freq, title, path, filename, show=True, save=False):
    """A helper function to plot the frequency power (Zipf law)"""
    fig, ax = plt.subplots()
    plt.plot(first_freq, second_freq, marker='*', fillstyle='none', linestyle='none', color='m')
    plt.ylabel("Frequency")
    plt.xlabel("Tokens")
    plt.xticks(rotation=35)
    ax.set_title(title)
    if save:
        plt.savefig(os.path.join(path,filename,".png"))
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


def plot_model_loss(out):
    metric = "accuracy"
    plt.figure()
    plt.plot(out.history[metric])
    plt.plot(out.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()



