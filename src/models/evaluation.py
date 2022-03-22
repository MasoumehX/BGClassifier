import os
import numpy as np
import pandas as pd
from tensorflow import keras
from models.cnn import train
from utils import load_dataset, compute_confusion_matrices
from metric import compute_acc_prec_rec_f1
from models.baseline import BaseLine
from plot import plot_model_loss


def prediction(model, model_name, xtest):
    preds = model.predict(xtest)
    if model_name.startswith('nn'):
        return np.argmax(preds, axis=1)
    return preds


def run_experiment(experiment_name=None,
                   group='head',
                    test_size=0.1,
                    model_name='nn_CNN',
                    task='multi',
                    features_type='norm',
                    metric_avg='macro',
                    read_path='./',
                    write_path='./'):

    target_classes = ['deictic', 'sequential', 'demarcative']
    print('-----------------Loading dataset --------------------------')
    print()
    dir_name = '/'.join([group, model_name, str(test_size), task, features_type])
    set_name = group + '_' + task + '_' + features_type + '_' + str(test_size) + '_' + model_name
    print('path: ', dir_name)
    print('filename: ', set_name)
    print()
    xtrain, ytrain, xtest, ytest, max_seq_len = load_dataset(set_name, os.path.join(read_path, dir_name))

    print(" -----------------------Training---------------------------")
    print()
    print('Model: ', model_name)
    print()
    if task == 'multi':
        n_classes = 3
        loss_func = "categorical_crossentropy"
        target_classes = ['deictic', 'sequential', 'demarcative']
    else:
        if task == 'bi_dem':
            target_classes = ['others', 'demarcative']
        if task == 'bi_dei':
            target_classes = ['others', 'deictic']
        if task == 'bi_seq':
            target_classes = ['others', 'sequential']
        n_classes = 2
        loss_func = "binary_crossentropy"

    if model_name.startswith('nn'):
        y_train = keras.utils.to_categorical(ytrain)
        y_test = keras.utils.to_categorical(ytest)
        model, history = train(xtrain, y_train, xtest, y_test,
                               input_shape=(max_seq_len, 4),
                               n_classes=n_classes,
                               drop_out=0.2,
                               special_value=99999,
                               lr=0.0001,
                               epochs=100,
                               batch_size=64,
                               loss_func=loss_func,
                               task=task)

        # plot loss
        plot_model_loss(history=history,
                        save=True,
                        show=False,
                        path=write_path + 'loss/',
                        filename=experiment_name)

    else:
        model = BaseLine(model=model_name)
        model.forward(xtrain, ytrain)

    print(" -----------------------Evaluation---------------------------")
    print()
    y_preds = prediction(model, model_name, xtest)
    acc, per, rec, f1 = compute_acc_prec_rec_f1(y_true=ytest, y_pred=y_preds, average=metric_avg)

    compute_confusion_matrices(y_true=ytest,
                               y_pred=y_preds,
                               path_to_save=write_path,
                               target_names=target_classes,
                               filename=experiment_name,
                               with_plot=True)

    return acc, per, rec, f1


def run_experiments(data_path='./', output_path='./'):
    groups = ['head', 'hands', 'head_hands', 'all']
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tasks = ['bi_dem', 'bi_seq', 'bi_dei', 'multi']
    model_names = ['base_SVM', 'base_RandomForest', 'nn_CNN']
    features_norms = ['norm', 'unnorm']

    # groups = ['head']
    # test_sizes = [0.1]
    # tasks = ['bi_dem']
    # model_names = ['nn_CNN']
    # features_norms = ['norm']
    metric_average = 'weighted'
    repeats = 5

    print()
    total_exp = len(groups) * len(test_sizes) * len(tasks) * len(model_names) * len(features_norms)
    print(total_exp, ' experiments to be run!')
    print()
    experiments = list()
    a_scores_all = list()
    p_scores_all = list()
    r_scores_all = list()
    f_scores_all = list()
    for group in groups:
        for test_size in test_sizes:
            for task in tasks:
                for model_name in model_names:
                    for features_norm in features_norms:
                        a_scores = list()
                        p_scores = list()
                        r_scores = list()
                        f_scores = list()
                        experiment_name = '-'.join([group, task, features_norm, str(test_size), model_name])
                        print('Experiment Name: ', experiment_name)
                        for i in range(repeats):
                            acc, per, rec, f1 = run_experiment(experiment_name=experiment_name,
                                                               group=group,
                                                               test_size=test_size,
                                                               model_name=model_name,
                                                               task=task,
                                                               metric_avg=metric_average,
                                                               features_type=features_norm,
                                                               read_path=data_path,
                                                               write_path=output_path)
                            a_scores.append(acc)
                            p_scores.append(per)
                            r_scores.append(rec)
                            f_scores.append(f1)

                        experiments.append(experiment_name)
                        a_scores_all.append(a_scores)
                        p_scores_all.append(p_scores)
                        r_scores_all.append(r_scores)
                        f_scores_all.append(f_scores)

    # saving the results in a data frame
    df_results = pd.DataFrame()
    df_results['experiment_name'] = experiments
    df_results['accuracies'] = a_scores_all
    df_results['precisions'] = p_scores_all
    df_results['recalls'] = r_scores_all
    df_results['f1s'] = f_scores_all
    return df_results


if __name__ == '__main__':
    df = run_experiments(data_path='/mnt/shared/people/masoumeh/MA/data/datasets/',
                    output_path='/mnt/shared/people/masoumeh/MA/results/evaluations/')
    print('Saving results...')
    pd.to_pickle(df,'/mnt/shared/people/masoumeh/MA/results/evaluations/df_evaluations.pickle')