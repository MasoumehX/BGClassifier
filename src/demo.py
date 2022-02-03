import time
import torch
import numpy as np

from models.baseline import BaseLine
# from models.nn import FeedForward, learning
from metric import compute_acc_acc5_f1_prec_rec
from torch.nn import BCEWithLogitsLoss, Linear, Module, ReLU, Tanh, NLLLoss





# print("train input shape:", train.name.unique().shape)
# print("test input shape", test.name.unique().shape)
# print(val.words.unique().shape)

# print(train[train.SemanticType == "demarcative"].name.unique().shape)
# print(train[train.SemanticType == "deictic"].name.unique().shape)
# print(train[train.SemanticType == "sequential"].name.unique().shape)
#
# print(test[test.SemanticType == "demarcative"].name.unique().shape)
# print(test[test.SemanticType == "deictic"].name.unique().shape)
# print(test[test.SemanticType == "sequential"].name.unique().shape)
#
# cols = ["x", "y", "point", "time", "frame", "fid", "classes", "label", "SemanticType"]
# print(train[cols])






# write_csv_file(dataset, path=root+"dataset_big.csv")
# write_csv_file(train, path=root+"train_big.csv")
# write_csv_file(test, path=root+"test_big.csv")
# write_csv_file(val, path=root+"val_3.csv")


# print("-"* 50, " Learning ", "-"*50)


# cues_col = ["x", "y", "time", "poi"]
# X_train = np.stack(train[cues_col].values)
# y_train = train["class"].values
#
# X_test = np.stack(test[cues_col].values)
# y_test = test["class"].values
#
# start = time.time()
# main(X_train=np.stack(X_train), y_train=y_train, X_test=np.stack(X_test), y_test=y_test)
# # end = time.time()

# ----------------------------------------------------

#
# def baseline_main(X_train, y_train, X_test, y_test, model="random forest"):
#
#     cues_col = ["x", "y", "time", "poi"]
#     X_train = np.stack(X_train[cues_col].values)
#     y_train = y_train["SemanticType"].values
#
#     X_test = np.stack(X_test[cues_col].values)
#     y_test = y_test["SemanticType"].values
#
#     start = time.time()
#     clf = BaseLine(model="random forest")
#     end = time.time()
#     print('total Time: {}'.format(end - start))
#     predictions = clf.predict(X_test)
#     print("Predicted Classes: ", list(set(predictions)))
#
#     print(" ========== RESULTS ===========")
#     compute_acc_acc5_f1_prec_rec(y_true=y_test, y_pred=predictions)
#
#
# def nn_main(X_train, y_train, X_test, y_test, model="FFN"):
#     # 0. get started
#     print("Begin predict student major ")
#     np.random.seed(1)
#     torch.manual_seed(1)
#
#     # 1. create Dataset and DataLoader objects
#     X_train = torch.Tensor(X_train).float()
#     y_train = torch.Tensor(y_train).float()
#     X_test = torch.Tensor(X_test).float()
#     y_test = torch.Tensor(y_test).float()
#
#     # 2. create neural network
#     model_torch = FeedForward(X_train.shape[1], hidden_size=10, activation=ReLU, n_classes=3)
#
#     # 3. train network
#     learning(model_torch, X_train, y_train, steps=1000, p_norm=2, lr=0.001, loss_function=NLLLoss())
#
#     # 4. evaluate model
#
#     # classes = ('sequential', 'deictic', 'demarcative')
#     # validate(nonlinear_model_torch, X_test, y_test, c=classes)
#
#     # 5. save model
#     # 6. make a prediction
