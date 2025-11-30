from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score,f1_score


def evaluate_model(x_train , y_train , x_test , y_test , classifier, report=False , cm=False):
    y_test_pred = classifier.predict(x_test)
    y_train_pred = classifier.predict(x_train)

    if report==False:
        train_acc = accuracy_score(y_train, y_train_pred)
        train_pre = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_pre = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        print("Training Set Evaluation:")
        print(f"Accuracy: {train_acc:.4f}")
        print(f"Precision: {train_pre:.4f}")
        print(f"Recall: {train_recall:.4f}")
        print(f"F1 Score: {train_f1:.4f}\n")

        print("Test Set Evaluation:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_pre:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1 Score: {test_f1:.4f}\n")
    
    else:
        print("Training Classification Report:")
        print(classification_report(y_train, y_train_pred, zero_division=0))

        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
    
    if cm==True:
        print("Confusion Matrix:")
        print("For training data:")
        print(confusion_matrix(y_train, y_train_pred))

        print("For test data:")
        print(confusion_matrix(y_test, y_test_pred))