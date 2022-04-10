from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class BaseLine:
    def __init__(self, model="base_svm"):
        if model == "SVM":
            self.clf = svm.NuSVC(kernel='rbf',
                                 gamma="scale",
                                 cache_size=1000,
                                 decision_function_shape='ovr',
                                 nu=0.01)
        elif model == "RandomForest":
            self.clf = RandomForestClassifier(n_estimators=70,
                                              criterion="entropy",
                                              random_state=1,
                                              min_samples_split=2,
                                              min_samples_leaf=150,
                                              n_jobs=-1)
    def forward(self, x, y):
        self.clf.fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)






