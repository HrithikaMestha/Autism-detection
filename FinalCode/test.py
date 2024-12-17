import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.tree import _tree

# Load the dataset
data = pd.read_csv('data.csv')
data.columns = data.columns.str.strip()
features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']
target = 'Class/ASD Traits'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, 'rf_model.pkl')
clf = joblib.load('rf_model.pkl')

# Extract rules from the Random Forest model
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, depth, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = current_rule + [(name, "<=", threshold)]
            right_rule = current_rule + [(name, ">", threshold)]
            recurse(tree_.children_left[node], depth + 1, left_rule)
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            rules.append((current_rule, tree_.value[node]))

    recurse(0, 1, [])
    return rules

def extract_rf_rules(rf, feature_names):
    all_rules = []
    for estimator in rf.estimators_:
        rules = tree_to_code(estimator, feature_names)
        all_rules.extend(rules)
    return all_rules

rules = extract_rf_rules(clf, features)

def apply_rules(rules, user_input):
    for rule, value in rules:
        match = True
        for feature, op, threshold in rule:
            if op == "<=":
                if user_input[feature] > threshold:
                    match = False
                    break
            elif op == ">":
                if user_input[feature] <= threshold:
                    match = False
                    break
        if match:
            prediction = np.argmax(value)
            return prediction
    return None

def predict_using_rules(user_answers, rules, feature_names):
    user_input = {feature: val for feature, val in zip(feature_names, user_answers)}
    prediction = apply_rules(rules, user_input)
    return "Autistic" if prediction == 1 else "Not Autistic"

# Streamlit app
st.title("Autism Screening Prediction")

st.write("""
### Please answer the following questions:
""")

# List of questions corresponding to A1, A2, ..., A10
questions = [
    "Does your child look at you when you call his/her name?",
    "How easy is it for your child to get along with others?",
    "Does your child point to indicate that s/he wants something?(e.g. a toy that is out of reach)",
    "Does your child point to share interest with you? (e.g. poin9ng at an interes9ng sight)",
    "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
    "Does your child follow where you’re looking?",
    "If you or someone else in the family is visibly upset, does your child show signs of wantng to comfort them? (e.g. stroking hair, hugging them)",
    "Does your child ever stare at nothing or wander with no purpose?",
    "Does your child use simple gestures? (e.g. wave goodbye) ",
    "Does your child stare at nothing with no apparent purpose?"
]

user_answers = []
for question in questions:
    user_input = st.selectbox(question, options=["Yes", "No"])
    user_answers.append(1 if user_input == "Yes" else 0)

if st.button('Predict'):
    result = predict_using_rules(user_answers, rules, features)
    st.write(f"The prediction is: **{result}**")
