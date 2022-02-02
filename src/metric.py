import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score


def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred, normalize=True)


def balanced_accuracy(y_true, y_pred):
    if y_pred.ndim == 1:
        return balanced_accuracy_score(y_true, y_pred)
    else:
        return balanced_accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))


def compute_acc_acc5_f1_prec_rec(y_true, y_pred):
    dic_model = dict()
    dic_model['acc'] = accuracy(y_true, y_pred)
    dic_model['balanced_accuracy'] = balanced_accuracy(y_true, y_pred)
    dic_model['precision_macro'] = precision_macro(y_true, y_pred)
    dic_model['recall_macro'] = recall_macro(y_true, y_pred)
    dic_model['f1-macro'] = f1_macro(y_true, y_pred)

    result = pd.DataFrame(dic_model, index=[0])

    return result