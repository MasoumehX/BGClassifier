import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.express as px
import seaborn as sns


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
                  filename=None,
                  iters=0):

    # Plot training & validation accuracy values
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    plt.title(title + "-" + filename)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig(path+filename+"_"+str(iters)+".png", bbox_inches='tight')
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


def plot_density(df_clean, path=None, filename=None):
    l = df_clean.columns.values
    number_of_columns = len(df_clean.columns)
    number_of_rows = len(l) - 1 / number_of_columns
    plt.figure(figsize=(3 * number_of_columns, 6 * number_of_rows))
    for i in range(0, len(l)):
        plt.subplot(int(number_of_rows) + 1, number_of_columns, i + 1)
        sns.distplot(df_clean[l[i]], kde=True, color='y')
    # plt.title('Kernel density plot')
    plt.xlabel('Key Body Points')
    plt.savefig(path + filename + ".png", bbox_inches='tight')



    df_head = df_clean[df_clean.point.isin([0, 15, 16, 17, 18])]
    df_hands = df_clean[df_clean.point.isin([2, 3, 4, 5, 6, 7])]
    df_head_hands = df_clean[df_clean.point.isin([0, 1, 15, 16, 17, 18, 2, 3, 4, 5, 6, 7])]
    df_all = df_clean



    df_deictic = df_clean[df_clean.SemanticType == 'deictic']
    df_sequential = df_clean[df_clean.SemanticType == 'sequential']
    df_demarcative = df_clean[df_clean.SemanticType == 'demarcative']



    df_head_dei = df_deictic[df_deictic.point.isin([0, 15, 16, 17, 18])]
    df_head_seq = df_sequential[df_sequential.point.isin([0, 15, 16, 17, 18])]
    df_head_dem = df_demarcative[df_demarcative.point.isin([0, 15, 16, 17, 18])]


    # features
    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_clean.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_clean.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_clean.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_clean.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    # fig.suptitle('Distribution of head points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_deictic.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_deictic.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_deictic.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_deictic.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    # fig.suptitle('Distribution of head points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords_deictic.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_head_dei.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_head_dei.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_head_dei.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_head_dei.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    fig.suptitle('Distribution of head points for deictic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords_head_deictic.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_hands.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_head_dei.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_head_dei.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_head_dei.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    fig.suptitle('Distribution of head points for deictic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords_head_deictic.png", bbox_inches='tight')
    plt.close()




    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_sequential.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_sequential.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_sequential.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_sequential.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    # fig.suptitle('Distribution of head points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords_sequential.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=4)
    sns.distplot(df_demarcative.x, kde=True, color='b', ax=ax[0], axlabel='x')
    sns.distplot(df_demarcative.y, kde=True, color='m', ax=ax[1], axlabel='y')
    sns.distplot(df_demarcative.nx, kde=True, color='g', ax=ax[2], axlabel='nx')
    sns.distplot(df_demarcative.ny, kde=True, color='r', ax=ax[3], axlabel='ny')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    ax[3].set(ylabel=None)
    # fig.suptitle('Distribution of head points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_coords_demarctive.png", bbox_inches='tight')
    plt.close()



    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=3)
    sns.distplot(df_head_dei, kde=True, color='b', ax=ax[0], axlabel='deictic')
    sns.distplot(df_head_seq, kde=True, color='m', ax=ax[1], axlabel='sequential')
    sns.distplot(df_head_dem, kde=True, color='g', ax=ax[2], axlabel='demarcative')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    fig.suptitle('Distribution of head points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_heads.png", bbox_inches='tight')
    plt.close()

    df_hands_dei = df_deictic[df_deictic.point.isin([2, 3, 4, 5, 6, 7])].point
    df_hands_seq = df_sequential[df_sequential.point.isin([2, 3, 4, 5, 6, 7])].point
    df_hands_dem = df_demarcative[df_demarcative.point.isin([2, 3, 4, 5, 6, 7])].point

    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=3)
    sns.distplot(df_hands_dei, kde=True, color='b', ax=ax[0], axlabel='deictic')
    sns.distplot(df_hands_seq, kde=True, color='m', ax=ax[1], axlabel='sequential')
    sns.distplot(df_hands_dem, kde=True, color='g', ax=ax[2], axlabel='demarcative')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    fig.suptitle('Distribution of hands points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_hands.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
    sns.distplot(df_head.point, kde=True, color='b', ax=ax[0, 0], axlabel='head')
    sns.distplot(df_hands.point, kde=True, color='r', ax=ax[0, 1], axlabel='hands')
    sns.distplot(df_head_hands.point, kde=True, color='g', ax=ax[1, 0], axlabel='upper body')
    sns.distplot(df_all.point, kde=True, color='m', ax=ax[1, 1], axlabel='whole body')
    ax[0, 0].set(ylabel=None)
    ax[0, 1].set(ylabel=None)
    ax[1, 0].set(ylabel=None)
    ax[1, 1].set(ylabel=None)
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_groups.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 9), nrows=1, ncols=3)
    sns.distplot(df_deictic.point, kde=True, color='r', ax=ax[0], axlabel='deictic')
    sns.distplot(df_sequential.point, kde=True, color='purple', ax=ax[1], axlabel='sequential')
    sns.distplot(df_demarcative.point, kde=True, color='c', ax=ax[2], axlabel='demarcative')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    ax[2].set(ylabel=None)
    fig.suptitle('Distribution of all points for semantic classes')
    fig.text(0.5, 0.04, 'points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_all_points.png", bbox_inches='tight')
    plt.close()

    # Deictic
    df_head_dei = df_deictic[df_deictic.point.isin([0, 15, 16, 17, 18])].point
    df_hands_dei = df_deictic[df_deictic.point.isin([2, 3, 4, 5, 6, 7])].point
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2)
    sns.distplot(df_head_dei, kde=True, color='b', ax=ax[0], axlabel='head')
    sns.distplot(df_hands_dei, kde=True, color='r', ax=ax[1], axlabel='hands')
    ax[0].set(ylabel=None)
    ax[0].set(ylabel=None)
    fig.suptitle('Key Body Points for Deictic Gestures')
    fig.text(0.5, 0.04, 'Points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_deictic.png", bbox_inches='tight')
    plt.close()

    # Sequential
    df_head_seq = df_sequential[df_sequential.point.isin([0, 15, 16, 17, 18])].point
    df_hands_seq = df_sequential[df_sequential.point.isin([2, 3, 4, 5, 6, 7])].point
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2)
    sns.distplot(df_head_seq, kde=True, color='b', ax=ax[0], axlabel='head')
    sns.distplot(df_hands_seq, kde=True, color='r', ax=ax[1], axlabel='hands')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    fig.suptitle('Key Body Points for Sequential Gestures')
    fig.text(0.5, 0.04, 'Points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_sequential.png", bbox_inches='tight')
    plt.close()

    # Demarcative
    df_head_dem = df_demarcative[df_demarcative.point.isin([0, 15, 16, 17, 18])].point
    df_hands_dem = df_demarcative[df_demarcative.point.isin([2, 3, 4, 5, 6, 7])].point
    fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2)
    sns.distplot(df_head_dem, kde=True, color='b', ax=ax[0], axlabel='head')
    sns.distplot(df_hands_dem, kde=True, color='r', ax=ax[1], axlabel='hands')
    ax[0].set(ylabel=None)
    ax[1].set(ylabel=None)
    fig.suptitle('Key Body Points for Demarcative Gestures')
    fig.text(0.5, 0.04, 'Points', ha='center')
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_density_demarcative.png", bbox_inches='tight')
    plt.close()


def plot_line_model_performance(dff, model_name='SVM', score_col='mean_f1'):
    new_labels = ['head', 'hands', 'upper body', 'whole body']
    old_labels = ['head', 'hands', 'head_hands', 'all']
    cnn_scores = dff[(dff.task == 'multi') & (dff.model_name == 'CNN')].groupby(['group']).best_f1.max().reindex(old_labels).values
    svm_scores = dff[(dff.task == 'multi') & (dff.model_name == 'SVM')].groupby(['group']).best_f1.max().reindex(old_labels).values
    forest_scores = dff[(dff.task == 'multi') & (dff.model_name == 'RandomForest')].groupby(['group']).best_f1.max().reindex(old_labels).values

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(new_labels, cnn_scores, label='cnn', marker='.')
    ax.plot(new_labels, svm_scores, label='svm', marker='.')
    ax.plot(new_labels, forest_scores, label='random forest', marker='.')

    ax.set(ylabel='F1 score')
    ax.set(xlabel='experiments')

    ax.legend()

    plt.savefig("/mnt/shared/people/masoumeh/MA/results/evaluations/plots/plot_results_experiments_multi.png", bbox_inches='tight')
    plt.close()



    dff_head_norm = dff[(dff.group == 'head') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'norm')]
    dff_head_unnorm = dff[(dff.group == 'head') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'unnorm')]

    dff_hands_norm = dff[(dff.group == 'hands') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'norm')]
    dff_hands_unnorm = dff[(dff.group == 'hands') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'unnorm')]

    dff_head_hands_norm = dff[(dff.group == 'head_hands') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'norm')]
    dff_head_hands_unnorm = dff[(dff.group == 'head_hands') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'unnorm')]

    dff_all_norm = dff[(dff.group == 'all') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'norm')]
    dff_all_unnorm = dff[(dff.group == 'all') & (dff.task == 'multi') & (dff.model_name == model_name) & (dff.isNormalized == 'unnorm')]

    fig, ax = plt.subplots(figsize=(12, 9), nrows=2, ncols=2)
    ax[0, 0].plot(dff_head_norm.testSize.tolist(), dff_head_norm[score_col].tolist(), label='norm', marker='.')
    ax[0, 0].plot(dff_head_unnorm.testSize.tolist(), dff_head_unnorm[score_col].tolist(), label='unnorm', marker='.')

    ax[0, 1].plot(dff_hands_norm.testSize.tolist(), dff_hands_norm[score_col].tolist(), label='norm', marker='.')
    ax[0, 1].plot(dff_hands_unnorm.testSize.tolist(), dff_hands_unnorm[score_col].tolist(), label='unnorm', marker='.')

    ax[1, 0].plot(dff_head_hands_norm.testSize.tolist(), dff_head_hands_norm[score_col].tolist(), label='norm', marker='.')
    ax[1, 0].plot(dff_head_hands_unnorm.testSize.tolist(), dff_head_hands_unnorm[score_col].tolist(), label='unnorm', marker='.')

    ax[1, 1].plot(dff_all_norm.testSize.tolist(), dff_all_norm[score_col].tolist(), label='norm', marker='.')
    ax[1, 1].plot(dff_all_unnorm.testSize.tolist(), dff_all_unnorm[score_col].tolist(), label='unnorm', marker='.')

    ax[0, 0].set(ylabel='score')
    ax[0, 1].set(ylabel=None)
    ax[1, 0].set(ylabel='score')
    ax[1, 1].set(ylabel=None)

    ax[0, 0].set(xlabel=None)
    ax[0, 1].set(xlabel=None)
    ax[1, 0].set(xlabel='test size')
    ax[1, 1].set(xlabel='test size')

    ax[0, 0].set(title='head points')
    ax[0, 1].set(title='hands point')
    ax[1, 0].set(title='upper body')
    ax[1, 1].set(title='whole body')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.savefig("/mnt/shared/people/masoumeh/MA/results/evaluations/plots/plot_results_" + model_name + "_multi.png", bbox_inches='tight')
    plt.close()

    # sns.distplot(df_head_dei, kde=True, color='b', ax=ax[0], axlabel='deictic')
    # sns.distplot(df_head_seq, kde=True, color='m', ax=ax[1], axlabel='sequential')
    # sns.distplot(df_head_dem, kde=True, color='g', ax=ax[2], axlabel='demarcative')
    # ax[0].set(ylabel=None)
    # ax[1].set(ylabel=None)
    # ax[2].set(ylabel=None)



def plot_catplot(df_clean):
    sns.set_theme(style="whitegrid")

    plt.figure()
    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df_clean, kind="bar",
        x="SemanticType", y="gesture frequency", hue="SemanticType",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Body mass (g)")
    g.legend.set_title("Distribution of Gestures")
    plt.savefig("/mnt/shared/people/masoumeh/MA/data/plots/density/plot_bar_gestures.png", bbox_inches='tight')
    plt.close()


def plot_evaluation_testsize(df):

    sns.set_theme(style="ticks")

    # Create a dataset with many short random walks
    rs = np.random.RandomState(4)
    pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
    pos -= pos[:, 0, np.newaxis]
    step = np.tile(range(5), 20)
    walk = np.repeat(range(20), 5)
    # df = pd.DataFrame(np.c_[pos.flat, step, walk],
    #                   columns=["position", "step", "walk"])

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="testSize", hue="testSize", palette="tab20c",
                         col_wrap=6, height=1.5)

    # Draw a horizontal line to show the starting point
    grid.refline(y=0, linestyle=":")

    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "step", "position", marker="o")

    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(5), yticks=[-3, 3],
             xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)

