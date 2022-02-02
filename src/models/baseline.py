from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BaseLine:
    def __init__(self, model="svm"):
        if model == "svm":
            self.clf = SVC(kernel='rbf', gamma="auto", cache_size=1000, decision_function_shape='ovr')
        elif model == "random forest":
            self.randomForest = RandomForestClassifier(n_estimators=70, criterion="entropy", random_state=1, min_samples_split=2, min_samples_leaf=150, n_jobs=-1)
        else:
            self.logisticReg = LogisticRegression(random_state=1, max_iter=1000, warm_start=False)

    def forward(self, x, y):
        return self.clf.fit(x, y)





