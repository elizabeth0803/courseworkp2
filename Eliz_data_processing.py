#Data Processing 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

#data explore

# Load the dataset
train_df = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split

#pd.set_option('display.max_columns', None)
#display(train_df)


train_df, valid_df = train_test_split(
    train_df,
    test_size = 0.2,
    random_state = 5059
)
#could do stratified split, particularly for CL data
def pipeline(df):

    labels = df['Status']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder()),
    ])
    
    num_attribs = ['N_Days', 'Bilirubin', 'Copper', 'Alk_Phos']
    cat_attribs = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders','Edema']

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs),
    ])

    df_prepared = full_pipeline.fit_transform(df)
    return df_prepared, labels

train_prepared, train_labels = pipeline(train_df)
valid_prepared, valid_labels = pipeline(valid_df)

#=================================================

tree_classifier = DecisionTreeClassifier(random_state=5059,)
                                        #min_samples_split = 100,
                                        # min_samples_leaf = 20,
                                        # max_depth = 25)
tree_classifier.fit(train_prepared, train_labels)
tree_predictions = tree_classifier.predict_proba(valid_prepared)[:,1]
tree_score = tree_classifier.predict_proba(valid_prepared)
#FPR_tree, TPR_tree, thresholds_tree = roc_curve(valid_labels, tree_predictions)
#auc_tree = roc_auc_score(valid_labels, tree_predictions)
#print('auc_tree', auc_tree)


#from sklearn.preprocessing import  LabelBinarizer#

#label_binarizer = LabelBinarizer.fit(train_labels)
#y_onehot_test = label_binarizer.transform(valid_labels)
#y_onehot_test.shape

#classes = ['D', 'C', 'CL']
#from sklearn.metrics import RocCurveDisplay

#for class_of_interest in classes:
#    class_id = np.flatnonzero(label_binarizer.classes == class_of_interest)[0]
#    display = RocCurveDisplay.from_predictions(
#        y_onehot_test[:, class_id],
#        tree_score[:, class_id],
#        name = f"{class_of_interest} vs the rest",
#        plot_chance_level = True)
#    _ = display.ax.et(
#        xlabel="FPR",
#        ylabel="TPR",
#        title = "One-vs-Rest Roc curves"
#    )

    
tree_predictions1 = tree_classifier.predict(valid_prepared)
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

matrix = confusion_matrix(valid_labels, tree_predictions1)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=tree_classifier.classes_)

disp.plot(cmap = 'binary')
plt.show()
print(matrix)

print(classification_report(valid_labels, tree_predictions1))



#plt.figure(figsize = (8,6))
#plt.plot(FPR_tree, TPR_tree, linewidth = 2, label = 'Tree')
#plt.plot([0,1], [0,1], color = 'k--')
#plt.axis([0,1,0,1])
#plt.xlabel('FPR (1 - specificity)')
#plt.ylabel('TPR (recall)', fontsize=16)
#plt.grid(True)
#plt.show()


