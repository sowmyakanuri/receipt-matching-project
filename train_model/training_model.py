import time
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score
from config_folder.config import Config

config = Config.get_instance().config

path_to_dataset = config['path_to_dataset']
path_to_save_file = config['path_to_save_model']

#Read the given csv file
df = pd.read_csv(path_to_dataset)
# df.head()
X=df.drop(["receipt_id","company_id","matched_transaction_id","success"],axis=1)
y=df["success"]

#Over sampling & Under sampling
over = SMOTENC(categorical_features=[3,4,6,7,9], random_state=123, k_neighbors=4)
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(X, y)

#PCA on data
# from sklearn import decomposition
# pca = decomposition.PCA()
# pca.fit(X)
# X = pca.transform(X)
# pca.explained_variance_ratio_

#train-test division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=1)

#Initial run on MLP
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X_train, y_train)

#Fine tuning the model
parameter_space = {
    'hidden_layer_sizes': [(5,2), (8,2),(10,2)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': [0.00001,0.001,0.01,0.1],
    'learning_rate': ['constant','adaptive'],
}
mlp = MLPClassifier(early_stopping=True,random_state=4)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3,scoring="f1")
clf.fit(X_train, y_train)

# Testing the model with test set
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print( precision_score(y_test, y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

file_to_save_model = path_to_save_file + "model_" + str(time.time())
joblib.dump(clf,file_to_save_model)


