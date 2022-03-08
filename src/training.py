from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def train(x_train, x_validation, y_train, y_validation):
    numeric_features = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    categorical_features = ["Sex", "Ticket", "Embarked"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    models = [('LR', LogisticRegression()),
              ('LDA', Pipeline([("to_dense", DenseTransformer()),
                  ("classifier", LinearDiscriminantAnalysis())])),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              (("RFC"), RandomForestClassifier()),
              ('NB', Pipeline([("to_dense", DenseTransformer()),
                  ("classifier", GaussianNB())])),
              ('SVM', SVC(gamma='auto'))]
    pipelines = []
    for model in models:
        pipeline = (model[0], Pipeline(
            steps=[("preprocessor", preprocessor),
                   ("classifier", model[1])]
        ))
        pipelines.append(pipeline)

    results = []
    names = []

    for name, pipeline in pipelines:
        k_fold = KFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(pipeline, x_train, y_train, cv=k_fold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)

    for name, cv_results in zip(names, results):
        print(f"{name}: {round(cv_results.mean(), 3)} ± {round(cv_results.std(), 3)}")

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    return x_train, y_train, x_validation, y_validation
