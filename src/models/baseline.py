
from utils import *
from metric import *
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import torch
import numpy as np
from tqdm import trange
from torch import Tensor
from sklearn.svm import SVC
from torch.nn import BCEWithLogitsLoss, Linear, Module, ReLU, Tanh, NLLLoss


class SVM():
    def __init__(self, kernel, degree, **keys):
        self.kernel = kernel
        self.degree = degree
        self.config = keys
        self.clf = SVC(kernel=self.kernel, degree=self.degree, coef0=self.config)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def forward(self, x, y):
        model = self.fit(x, y)
        return model.predict(x, y)


class NonlinearFFN(torch.nn.Module):
    def __init__(self, n_features: int, n_labels: int, hidden_size: int = 10, activation=ReLU):
        """Construct a feed-forward neural network with the given number of
           input features, hidden size, and activation function (non-linear
           function)."""
        super(NonlinearFFN, self).__init__()
        self.hid1 = Linear(n_features, hidden_size)
        self.hid2 = Linear(hidden_size, hidden_size)
        self.non_linear = activation()
        self.output = Linear(hidden_size, n_labels)

        # initialization the weights and bias
        # torch.nn.init.xavier_uniform_(self.hid1.weight)
        # torch.nn.init.zeros_(self.hid1.bias)
        # torch.nn.init.xavier_uniform_(self.hid2.weight)
        # torch.nn.init.zeros_(self.hid2.bias)
        # torch.nn.init.xavier_uniform_(self.output.weight)
        # torch.nn.init.zeros_(self.output.bias)

    def classify(self, x: Tensor) -> Tensor:
        """
        Classify instances, returns the class label.
        Given an `x` of shape `[n_instances, n_features]`, returns the labels with shape
        `[n_instances]`.
        """

        m = torch.nn.LogSoftmax(dim=1)
        probs = m(self.forward(x))
        return torch.argmax(probs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the raw activations (wx+b) for all instances.
        Given an `x` of shape `[n_instances, n_features]`, returns activations with shape
        `[n_instances]`.
        """
        z = self.hid1(x)
        z = self.hid2(z)
        act = self.non_linear(self.hid2(z))
        out = self.output(act)
        return out.squeeze()


def optimize(model: NonlinearFFN, x: Tensor, y: Tensor, loss_function: Module = BCEWithLogitsLoss(), steps=10000,
             p_norm: int = 2, norm_scale: float = 0.05, lr: float = 0.1, weight_decay=0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    norms = []
    running_losses = []

    for step in trange(steps, desc="optimizing model"):
        optimizer.zero_grad()
        model_params = torch.nn.utils.parameters_to_vector(model.parameters())
        loss = loss_function(model(x), y.type(torch.long))
        if p_norm:
            if p_norm == 1:
                loss += norm_scale * model_params.norm(p_norm)
            elif p_norm == 2:
                loss += norm_scale * (model_params ** 2).sum()
            else:
                raise ValueError("Unknown norm, use '1' or '2'")
        loss.backward()
        optimizer.step()
        norms.append(float(model_params.norm()))
        running_losses.append(loss.detach().numpy())
        with torch.no_grad():
            print(model.classify(x))
            acc = model.classify(x).eq(y).to(torch.float).mean()
            print(f"Step: {step}, loss: {loss}, acc: {acc}")
    return np.array(running_losses)


def validate(model: NonlinearFFN, x: Tensor, y: Tensor, c: tuple):
    correct_pred = {classname: 0 for classname in c}
    total_pred = {classname: 0 for classname in c}
    predictions = model.classify(x)
    y_validate = y.detach().numpy().astype(int)

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
    total_correct_pred = torch.where(predictions == y, 1, 0).sum()
    overall_acc = 100 * float(total_correct_pred) / predictions.shape[0]
    print("Overall accuracy is: {:.1f} %".format(overall_acc))


def main(X_train, y_train, X_test, y_test):
    # 0. get started
    print("Begin predict student major ")
    np.random.seed(1)
    torch.manual_seed(1)

    # 1. create Dataset and DataLoader objects
    X_train = torch.Tensor(X_train).float()
    y_train = torch.Tensor(y_train).float()
    X_test = torch.Tensor(X_test).float()
    y_test = torch.Tensor(y_test).float()


    # 2. create neural network
    nonlinear_model_torch = NonlinearFFN(X_train.shape[1], hidden_size=10, activation=Tanh, n_labels=3)

    # 3. train network
    optimize(nonlinear_model_torch, X_train, y_train, steps=200, p_norm=2, lr=0.001, loss_function=NLLLoss())

    # 4. evaluate model
    classes = ('sequential', 'deictic', 'demarcative')
    validate(nonlinear_model_torch, X_test, y_test, c=classes)

    # 5. save model
    # 6. make a prediction
    print("End predict student major demo ")



clf = RandomForestClassifier(n_estimators=70, criterion="entropy", random_state=1, min_samples_split=2,
                             min_samples_leaf=150, n_jobs=-1)
# --------------------------- Results ----------------------------#
# Data set: 3
# total Time: 3.418243408203125
# Predicted Classes:  ['demarcative', 'sequential', 'deictic']

# Accuracy:  53.23
# precision macro:  36.04
# recall macro:  33.88
# F1 macro:  26.85

# -----------------------------------------------------------------#


clf = LogisticRegression(random_state=1, max_iter=1000, warm_start=False)

# --------------------------- Results ----------------------------#
# Data set: 3
# total Time: 4.900315761566162
# Predicted Classes:  ['sequential', 'deictic', 'demarcative']

# Accuracy:  53.96
# precision macro:  18.38
# recall macro:  33.18
# F1 macro:  23.38

# -----------------------------------------------------------------#


# clf = SVC(kernel='rbf', gamma="auto", cache_size=1000, decision_function_shape='ovr')
# --------------------------- Results ----------------------------#
# Data set: 3
# total Time: 1009.324230670929
# Predicted Classes:  ['deictic', 'sequential', 'demarcative']

# Accuracy:  53.45
# precision macro:  33.61
# recall macro:  33.14
# F1 macro:  24.12

# -----------------------------------------------------------------#





# # clf = svm.NuSVC(kernel='rbf', gamma="auto", cache_size=1000, decision_function_shape='ovr')
# clf = SVC(kernel='rbf', degree=3, C=1, cache_size=4000)


root = "/home/masoumeh/Desktop/MasterThesis/Data/"
train = read_csv_file(path=root+"train_3.csv")
test = read_csv_file(path=root+"test_3.csv")
# val = read_csv_file(path=root+"val.csv")

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

cues_col = ["x", "y", "time", "poi"]
X_train = np.stack(train[cues_col].values)
y_train = train["classes"].values

X_test = np.stack(test[cues_col].values)
y_test = test["classes"].values

start = time.time()
clf = clf.fit(X_train, y_train)
end = time.time()
print('total Time: {}'.format(end - start))
predictions = clf.predict(X_test)
print("Predicted Classes: ", list(set(predictions)))
# probabilities = clf.predict_proba(X_test)
# print("Probabilities: ", probabilities)


print(" ========== RESULTS ===========")
accuracy = accuracy(y_test, predictions)
recall = recall_macro(y_test, predictions)
precision = precision_macro(y_test, predictions)
f1 = f1_macro(y_test, predictions)
print('Accuracy: ', "%.2f" % (accuracy*100))
print('precision macro: ', "%.2f" % (precision*100))
print('recall macro: ', "%.2f" % (recall*100))
print('F1 macro: ', "%.2f" % (f1*100))


