from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_model
from data_splitting import split_data


data = pd.read_csv('./data/processed_data.csv')
data.drop(columns=['year', 'month', 'day', 'hour', 'minute'], inplace=True)
# data splitting
x_train, x_test, y_train, y_test = split_data(data, target_col='target', test_size=0.3, random_state=42, stratify_col='target', data_split_type="random")
# data scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# classifier = RandomForestClassifier(n_estimators=12, random_state=42)
# classifier = GradientBoostingClassifier(n_estimators=20, random_state=42)
classifier = LogisticRegression(random_state=42, max_iter=400)
classifier.fit(x_train, y_train)

evaluate_model(x_train, y_train, x_test, y_test, classifier, report=True, cm=True)