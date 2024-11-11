#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
'''
AdaBoost implements boosting using a regular decision tree based classifier. For the decision tree classifier, we can use sklearn.tree.DecisionTreeClassifier.

- Create a class (AdaBoostClassifier ?)
- In the ctor pass a parameter for number of boosting rounds i.e. number of internal trees we will use and store it
- In the ctor also create six blank lists, to store the predictors (the internal trees), the predictor weights, the training accuracy, the test accuracy, the training loss and the test loss
- Create a function called fit() in the class and pass in the X and y data
    - Initialize a weight vector (numpy array) of dimension (num_samples,). Fill it with initial values 1/num_samples. See np.full() function for this.
    - Loop for the number of boosting rounds
        1. create a DecisionTreeClassifier model (predictor), and fit it with the X,y data and the sample_weight paramter set to the weight_vector
        2. predict using this predictor and the X data
        3. calculate the weighted error rate = dot(weight_vector.T, (predictions != y)) # be careful to change the predictions != y to int - it will be all bools by default, see astype() function 
        4. calculate the predictor weight = 0.5 * ln((1 - weighted error rate)/weighted error rate)
        5. update the weight vector in two steps
            - weight vector = weight vector * np.exp(-1 * predictor weight * predictions * weight vector) # note the * and not dot. These are by element multiplication
            - weight vector = weight vector / sum(weight vector) # normalization step
        6. append the predictor to the predictor list and the predictor weight to the predictor weight list
        7. calculate accuracy score (predictions from step 2, y) and log_loss(predictions from step 2, y, labels = [0,1]) and append them to the train accuracy list and the train loss list
- Create a function called predict(self, X, y) 
    - Create predictions using the X, y and each predictor you saved in step 6.
    - Create weighted predictions for each prediction by multiplying each prediction by the corresponding weight from the prediction weight list in step 6
    - Create accuracy scores for each prediction using the X, y
    - Create log loss for each prediction using X, y, lables = [0,1]
    - Try to do these steps using list comprehension as they all produce lists

- Prep and split the data
- Create a AdaBoostClassifier object (from the class you wrote) with 500 boosting rounds
- Call predict once on it using train data and once using test data and calculate accuracy - these are your final accuracy values
- Plot the train loss and the test loss on the same plot against a X axis range(500)
- Look at the test accuracy list variable on the classifier object, and find out the argmax. The value there is the best accuracy 
you could have gotten, if you had stopped at that boosting round. 

'''
class AdaBoostClassfier:
    def __init__(self, num_of_trees = 10, max_depth = None, min_samples_split = 2):
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.predictors = []
        self.predictor_weights = []
        self.train_log_loss = []
        self.test_log_loss = []
        self.train_accuracy = []
        self.test_accuracy = []


    def fit(self, X: np.ndarray, y: np.ndarray):
        weight_vector = np.full((X.shape[0],),1/X.shape[0])
        for tree_idx in range(self.num_of_trees):
            tree_clf = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split,random_state=1234).fit(X,y,sample_weight = weight_vector)
            
            predictions = tree_clf.predict(X)
            # calculate weighted error rate
            weighted_error_rate = np.dot(weight_vector.T, (predictions != y).astype(int))
            
            # calculate predictor weight
            predictor_weight = 0.5 * np.log((1 - weighted_error_rate)/weighted_error_rate)
            
            # update weights
            weight_vector = weight_vector * np.exp(-1*predictor_weight*predictions*y)
            weight_vector = weight_vector / np.sum(weight_vector)
            self.predictors.append(tree_clf)
            self.predictor_weights.append(predictor_weight)
            self.train_accuracy.append(accuracy_score(predictions,y))
            self.train_log_loss.append(log_loss(predictions,y,labels=[0,1]))
        return self

        
    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # should be num_of_trees x X.shape[0]
        preds = [predictor.predict(X) for predictor in self.predictors]
        weighted_predictions = np.array([predictions*self.predictor_weights[idx] for idx, predictions in enumerate(preds)])
        self.test_accuracy = [accuracy_score(predictions,y) for predictions in preds]
        self.test_log_loss = [log_loss(predictions,y,labels=[0,1]) for predictions in preds]
        return (np.sum(weighted_predictions,axis=0) > 0).astype(int)
          


#%%
df_data = pd.read_csv("C:/pix/ml/hw/decisiontree/train.csv")
# drop columns that are meaningless for this
df_data = df_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1).dropna()
# get the X data
df_X = df_data.drop("Survived",axis=1)
# get the y data
df_y = df_data["Survived"]
# one hot encode categoricals
df_X = pd.get_dummies(df_X,columns=["Pclass","Sex","Embarked"])
# make all X data numeric
df_X = df_X.astype(float)
# split
train_X,test_X,train_y,test_y = train_test_split(df_X,df_y,test_size=0.20,stratify=df_y,shuffle=True,random_state=1234)

#%%
adaBoost_clf = AdaBoostClassfier(num_of_trees=20, max_depth=128,min_samples_split=8).fit(train_X.values,train_y.values)
#%%
predictions_train = adaBoost_clf.predict(train_X.values,train_y.values)
predictions_test = adaBoost_clf.predict(test_X.values,test_y.values)
# %%
print(accuracy_score(predictions_train,train_y.values))
print(accuracy_score(predictions_test,test_y.values))
# %%
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(0,20),adaBoost_clf.train_log_loss,adaBoost_clf.test_log_loss)


# %%
