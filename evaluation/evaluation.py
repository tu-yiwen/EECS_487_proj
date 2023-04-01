"""Here are some evaluation methods that can be used in our project."""
from sklearn.metrics import f1_score, accuracy_score

def evaluate(predictions,labels):
    f1_score_macro = f1_score(labels, predictions, average='macro')
    f1_score_micro = f1_score(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)
    return " macro f1 score: "+str(f1_score_macro)+" micro f1 score: "+str(f1_score_micro)+" accuracy: "+str(accuracy)

    