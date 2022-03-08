from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def predict(x_train, x_validation, y_train, y_validation=None, save_results=False):
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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier())])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_validation)

    if y_validation is not None:
        print(accuracy_score(y_validation, predictions))
        print(confusion_matrix(y_validation, predictions))
        print(classification_report(y_validation, predictions))

    output = pd.DataFrame({'PassengerId': x_validation.PassengerId, 'Survived': predictions})

    if save_results:
        output.to_csv("data/submission.csv", index=False)