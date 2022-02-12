from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from metric import compute_acc_acc5_f1_prec_rec
from data import *
from utils import *


class BaseLine:
    def __init__(self, model="svm"):
        if model == "svm":
            self.clf = svm.NuSVC(kernel='rbf', gamma="auto", cache_size=1000, decision_function_shape='ovr')
        elif model == "random forest":
            self.clf = RandomForestClassifier(n_estimators=70, criterion="entropy", random_state=1, min_samples_split=2, min_samples_leaf=150, n_jobs=-1)
        elif model == "logistic":
            self.clf = LogisticRegression(random_state=1, max_iter=1000, warm_start=False, tol=1.0e-8, solver='lbfgs')
    def forward(self, x, y):
        self.clf.fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)


def run(X_train, y_train, X_test, y_test, base_model):
    print(" ---------------------------------------------------------")
    model = BaseLine(model=base_model)
    model.forward(X_train, y_train)
    predictions = model.predict(X_test)
    print(" ----------------  Model: ", base_model, " ----------------")
    return compute_acc_acc5_f1_prec_rec(y_true=y_test, y_pred=predictions)



if __name__ == '__main__':

    # On Server
    path = "/data/home/masoumeh/Data/"
    df_data = read_csv_file(path+"dataset_big_clean.csv")
    X_data, y_data = create_data_for_training(df_data, with_zero_pad=True, model="baseline")
    X_train, ytrain, X_test, ytest = split_train_test(X_data, y_data)
    run(X_train, ytrain, X_test, ytest, base_model="svm")
    run(X_train, ytrain, X_test, ytest, base_model="random forest")
    run(X_train, ytrain, X_test, ytest, base_model="logistic")
