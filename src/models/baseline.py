from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BaseLine:
    def __init__(self, model="base_svm"):
        if model == "base_SVM":
            self.clf = svm.NuSVC(kernel='rbf',
                                 gamma="scale",
                                 cache_size=1000,
                                 decision_function_shape='ovr',
                                 nu=0.01)
        elif model == "base_RandomForest":
            self.clf = RandomForestClassifier(n_estimators=70,
                                              criterion="entropy",
                                              random_state=1,
                                              min_samples_split=2,
                                              min_samples_leaf=150,
                                              n_jobs=-1)
        elif model == "base_Logistic":
            self.clf = LogisticRegression(random_state=1,
                                          max_iter=100000,
                                          warm_start=True,
                                          tol=1.0e-8,
                                          solver='lbfgs')
    def forward(self, x, y):
        self.clf.fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)






