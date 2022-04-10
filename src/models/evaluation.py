import os
import numpy as np
import pandas as pd
from numpy import mean, std
from tensorflow import keras
from models.cnn import train
from utils import load_dataset, compute_confusion_matrices
from metric import compute_acc_prec_rec_f1
from models.baseline import BaseLine
from plot import plot_model_loss


def prediction(model, model_class, xtest):
    preds = model.predict(xtest)
    if model_class.startswith('nn'):
        return np.argmax(preds, axis=1)
    return preds


def run_experiment(experiment_name=None,
                    group='head',
                    test_size=0.1,
                    model_class='nn',
                    model_name='CNN',
                    task='multi',
                    features_type='norm',
                    metric_avg='macro',
                    read_path='./',
                    write_path='./',
                    iters=0):

    target_classes = ['deictic', 'sequential', 'demarcative']
    print('-----------------Loading dataset --------------------------')
    print()

    dir_name = '/'.join([group, model_class, str(test_size), task, features_type])
    set_name = group + '_' + task + '_' + features_type + '_' + str(test_size) + '_' + model_class
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

    if model_class.startswith('nn'):
        y_train = keras.utils.to_categorical(ytrain)
        y_test = keras.utils.to_categorical(ytest)
        model, history = train(xtrain, y_train, xtest, y_test,
                               input_shape=(max_seq_len, 4),
                               n_classes=n_classes,
                               drop_out=0.2,
                               special_value=99999,
                               lr=0.0001,
                               epochs=10,
                               batch_size=64,
                               loss_func=loss_func,
                               task=task)

        # plot loss
        plot_model_loss(history=history,
                        save=True,
                        show=False,
                        path=write_path + 'loss/',
                        filename=experiment_name,
                        iters=iters)

    else:
        model = BaseLine(model=model_name)
        model.forward(xtrain, ytrain)

    print(" -----------------------Evaluation---------------------------")
    print()
    y_train_preds = prediction(model, model_class, xtrain)
    acc_train, per_train, rec_train, f1_train = compute_acc_prec_rec_f1(y_true=ytrain, y_pred=y_train_preds, average=metric_avg)
    y_preds = prediction(model, model_class, xtest)
    acc, per, rec, f1 = compute_acc_prec_rec_f1(y_true=ytest, y_pred=y_preds, average=metric_avg)

    compute_confusion_matrices(y_true=ytest,
                               y_pred=y_preds,
                               path_to_save=write_path,
                               target_names=target_classes,
                               filename=experiment_name,
                               with_plot=True,
                               iters=iters)

    return acc, per, rec, f1, acc_train, per_train, rec_train, f1_train


def run_experiments(groups,
                    test_sizes,
                    tasks,
                    model_names,
                    model_class,
                    features_norms,
                    metric_average,
                    repeats,
                    data_path='./',
                    output_path='./'):
    print()
    total_exp = len(groups) * len(test_sizes) * len(tasks) * len(model_names) * len(features_norms)
    print(total_exp, ' experiments to be run!')
    print()
    experiments = list()
    a_scores_all = list()
    p_scores_all = list()
    r_scores_all = list()
    f_scores_all = list()

    a_scores_train_all = list()
    p_scores_train_all = list()
    r_scores_train_all = list()
    f_scores_train_all = list()

    for group in groups:
        for test_size in test_sizes:
            for task in tasks:
                for model_name in model_names:
                    for features_norm in features_norms:
                        a_scores_tt = list()
                        p_scores_tt = list()
                        r_scores_tt = list()
                        f_scores_tt = list()

                        a_scores = list()
                        p_scores = list()
                        r_scores = list()
                        f_scores = list()

                        experiment_name = '-'.join([group, task, features_norm, str(test_size), model_name])
                        print('Experiment Name: ', experiment_name)
                        for i in range(repeats):
                            acc, per, rec, f1, acc_tt, per_tt, rec_tt, f1_tt = run_experiment(experiment_name=experiment_name,
                                                                                               group=group,
                                                                                               test_size=test_size,
                                                                                               model_class=model_class,
                                                                                               model_name=model_name,
                                                                                               task=task,
                                                                                               metric_avg=metric_average,
                                                                                               features_type=features_norm,
                                                                                               read_path=data_path,
                                                                                               write_path=output_path,
                                                                                               iters=i)

                            # test
                            a_scores.append(acc)
                            p_scores.append(per)
                            r_scores.append(rec)
                            f_scores.append(f1)

                            # train
                            a_scores_tt.append(acc_tt)
                            p_scores_tt.append(per_tt)
                            r_scores_tt.append(rec_tt)
                            f_scores_tt.append(f1_tt)

                        experiments.append(experiment_name)
                        # test
                        a_scores_all.append(a_scores)
                        p_scores_all.append(p_scores)
                        r_scores_all.append(r_scores)
                        f_scores_all.append(f_scores)

                        # train
                        a_scores_train_all.append(a_scores)
                        p_scores_train_all.append(p_scores)
                        r_scores_train_all.append(r_scores)
                        f_scores_train_all.append(f_scores)

    df_results = pd.DataFrame()
    df_results['experiment_name'] = experiments
    df_results['accuracies'] = a_scores_all
    df_results['precisions'] = p_scores_all
    df_results['recalls'] = r_scores_all
    df_results['f1s'] = f_scores_all
    return df_results


def process_evaluation(df_results):
    """ A helper function to process the evaluations"""

    df_results['mean_acc'] = df_results.accuracies.apply(lambda x: mean(x))
    df_results['mean_precision'] = df_results.precisions.apply(lambda x: mean(x))
    df_results['mean_recall'] = df_results.recalls.apply(lambda x: mean(x))
    df_results['mean_f1'] = df_results.f1s.apply(lambda x: mean(x))

    df_results['std_acc'] = df_results.accuracies.apply(lambda x: std(x))
    df_results['std_precision'] = df_results.precisions.apply(lambda x: std(x))
    df_results['std_recall'] = df_results.recalls.apply(lambda x: std(x))
    df_results['std_f1'] = df_results.f1s.apply(lambda x: std(x))

    df_results['best_acc'] = df_results.accuracies.apply(lambda x: max(x))
    df_results['best_precision'] = df_results.precisions.apply(lambda x: max(x))
    df_results['best_recall'] = df_results.recalls.apply(lambda x: max(x))
    df_results['best_f1'] = df_results.f1s.apply(lambda x: max(x))

    df_results['group'] = df_results.experiment_name.apply(lambda x: x.split('-')[0])
    df_results['task'] = df_results.experiment_name.apply(lambda x: x.split('-')[1])
    df_results['isNormalized'] = df_results.experiment_name.apply(lambda x: x.split('-')[2])
    df_results['testSize'] = df_results.experiment_name.apply(lambda x: x.split('-')[3])
    df_results['model_name'] = df_results.experiment_name.apply(lambda x: x.split('-')[-1])

    return df_results


def read_evaluation(df_eval, model_name='svm', group='head', task='multi'):
    """ A helper function to filter experiments based on different clues."""

    dd = df_eval[(df_eval.model_name == model_name) & (df_eval.group == group)]
    dd = dd[dd.task == task]
    return dd


if __name__ == '__main__':
    groups = ['head', 'hands', 'head_hands', 'all']
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tasks = ['multi', 'bi_dem', 'bi_seq', 'bi_dei']
    model_class = 'base'
    # model_names = ['CNN']
    model_names = ['SVM', 'RandomForest']
    features_norms = ['norm', 'unnorm']
    n_iters = 1
    metric_average = 'weighted'
    path_data='/mnt/shared/people/masoumeh/MA/data/datasets/'
    path_output = '/mnt/shared/people/masoumeh/MA/results/evaluations/base/'
    df = run_experiments(groups=groups,
                         test_sizes=test_sizes,
                         tasks=tasks,
                         model_names=model_names,
                         model_class=model_class,
                         features_norms=features_norms,
                         metric_average=metric_average,
                         repeats=n_iters,
                         data_path=path_data,
                         output_path=path_output)
    print('Saving results...')

    df = process_evaluation(df_results=df)
    pd.to_pickle(df, '/mnt/shared/people/masoumeh/MA/results/evaluations/df_'+model_class+'_evaluations.pickle')

    # df_filtered = read_evaluation(df_eval=df_cnn, model_name='nn_CNN', group='head', task='multi')
