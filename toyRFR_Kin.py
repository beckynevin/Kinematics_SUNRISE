#!/usr/bin/env python


#Usage: a simple implementation of an RFR on toy data



import matplotlib
matplotlib.use('Agg')
import numpy as np
#from astropy.stats import biweight_scale
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor


########################### ADJUSTABLE PARAMETERS ###########################
Nfolds = 10
Ndat = 5000

#############################################################################
 




    
def testAndTrainIndices(test_fold, Nfolds, folds):
    
    print('finding test and train indices...')

    train_folds = np.delete(np.arange(Nfolds), test_fold)

    test_ind   = [i for i in range(len(folds)) if folds[i]==test_fold]
    train_ind  = [i for i in range(len(folds)) if folds[i] in train_folds]

    return test_ind, train_ind

def main():

    add_on='fg3_m10_degraded'
    run = 'fg3_m10'

    #load the data set
    print('loading data set...')
    import pandas as pd
    df = pd.io.parsers.read_table(
    filepath_or_buffer='LDA_kin_kurt_'+run+'_degraded.txt',#'_view_all.txt',#filepath_or_buffer='LDA_img_ratio_'+str(run)+'_early_late_all_things.txt',#'_view_all.txt',
    header=[0],
    sep='\t'
    )
    
    for j in range(len(df)):
        if df[['Myr']].values[j][0]<0.39:#df[['Myr']].values[i][0]
            df.at[j,'class label']=0
        if add_on[:7]=='fg3_m12' and (df[['Myr']].values[j][0]-2.15) > 0.5:#0.39+0.1:#was 0.39
            df.at[j,'class label']=0
        if add_on[:7]=='fg1_m13' and (df[['Myr']].values[j][0]-2.74) > 0.5:#2.74-2.25+2.74:#was 0.39
            df.set_value(j,'class label',0)
        if add_on[:7]=='fg3_m13' and (df[['Myr']].values[j][0]-2.59) > 0.5:#2.64+0.5
            df.set_value(j,'class label',0)
        if add_on[:7]=='fg3_m15' and (df[['Myr']].values[j][0]-3.72) > 0.5:
            df.set_value(j,'class label',0)
        if add_on[:7]=='fg3_m10' and (df[['Myr']].values[j][0]-9.17) > 0.5:
            df.set_value(j,'class label',0)
        if add_on[:5]=='major':
            #then sort by image name
            if df[['Image']].values[j][0][:7]=='fg3_m12' and (df[['Myr']].values[j][0]-2.15) > 0.5:#q0.5_fg0.3
                df.set_value(j,'class label',0)
            if df[['Image']].values[j][0][:7]=='fg1_m13' and (df[['Myr']].values[j][0]-2.59) > 0.5:
                df.set_value(j,'class label',0)
            if df[['Image']].values[j][0][:7]=='fg3_m13' and (df[['Myr']].values[j][0]-2.74) > 0.5:
                df.set_value(j,'class label',0)
        if add_on[:5]=='minor':
            #then sort by image name
            if df[['Image']].values[j][0][:7]=='fg3_m15' and (df[['Myr']].values[j][0]-3.72) > 0.5:
                df.set_value(j,'class label',0)
            if df[['Image']].values[j][0][:7]=='fg3_m10' and (df[['Myr']].values[j][0]-9.17) > 0.5:
                df.set_value(j,'class label',0)
            
    
    print(df)
    
    #dat = np.load('LDA_kin_fg3_m12_degraded_cleaned.txt')
    
    features =df[['Delta PA','v_asym','s_asym', 'resids','lambda_r', 'epsilon','A','A_2','deltapos','deltapos2','nspax','re',
        'meanvel','varvel','skewvel','kurtvel',
        'meansig','varsig','skewsig','kurtsig']].values
    #,'nspax','re'
    Nfeatures = len(features[0])
    
    #dat['features']#.reshape(-1,1)
    labels = df[['class label']].values
    #dat['labels']#.reshape(-1,1)
    

    '''#make fake data labels, ranging from 0 -100.
    labels = np.random.uniform(low=0, high=100, size=Ndat)

    #five features
    features = np.zeros((Ndat, 5), 'f')
    #features 0-3 correlate with the labels, but in a noisy way
    features[:, 0] = labels**1.4         + np.random.normal(scale=10,   size=Ndat)
    features[:, 1] = 10 * labels         + np.random.normal(scale=20,   size=Ndat)
    features[:, 2] = np.sin(labels/10.)  + np.random.normal(scale=0.05, size=Ndat)
    features[:, 3] = 1.1**labels         + np.random.normal(scale=20,   size=Ndat)
    #but feature 4 is absolute garbage:
    features[:, 4] = np.random.normal(scale=10, size=Ndat)
    
    '''

    #assign folds.  
    #There are no correlations we need to worry about, so we can do the simplest thing:
    folds = np.arange(len(labels))%Nfolds
    
    
    #Test on fold 0, train on the remaining folds:
    test_ind, train_ind = testAndTrainIndices(test_fold = 0, Nfolds = Nfolds, folds=folds)
    
    #divide features and labels into test and train sets:
    test_features = features[test_ind]
    test_labels   = labels[test_ind]
   
    train_features  = features[train_ind]
    train_labels    = labels[train_ind]

    print('training fold 0')
    #make a random forest model:
    model = RandomForestRegressor(max_depth=10, random_state=42)
    model.fit(train_features, train_labels)

    print('predicting...')
    # Predict on new data
    preds = model.predict(test_features)
    
    #print out the first few mass predictions to see if they make sense:
    for h in range(10):
        print(test_labels[h], preds[h])


    
    x = np.linspace(min([np.min(test_labels), np.min(preds)]), \
                        max([np.max(test_labels), np.max(preds)]), 100)
    plt.plot(x, x, c='k', ls='--')
    plt.scatter(test_labels, preds, s=3, alpha=0.3)
    plt.xlabel('truth')
    plt.ylabel('prediction')
    plt.savefig('rfrscatter.pdf')
    plt.clf()

    # rank feature importance:
    print('ranking feature importances...')
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(Nfeatures), importances[indices], yerr=std[indices], align="center", color='orange')
    plt.xticks(range(Nfeatures), indices)
    plt.xlim([-1, Nfeatures])
    plt.savefig('feature_importance_'+str(run)+'_gauss.pdf')
    
    

if __name__ == "__main__":
    main()
