#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:52:08 2019

@author: monika

parameter selection
"""

from sklearn.model_selection import GridSearchCV

param_1 = {'n_estimators':[10, 50, 100, 200]}
GS_1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,
                                  max_depth = 3, max_features = 'auto', subsample = 0.8), 
                       param_grid = param_1, scoring = 'f1_macro', iid = False, cv = 3)
GS_1.fit(x_train, y_train)
GS_1.grid_scores_, GS_1.best_params_, GS_1.best_score_

param_1 = {'n_estimators':[5, 10, 20, 25, 30]}
GS_1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate = 0.1,
                                  max_depth = 3, max_features = 'auto', subsample = 0.8), 
                       param_grid = param_1, scoring = 'f1_macro', iid = False, cv = 3)
GS_1.fit(x_train, y_train)
GS_1.grid_scores_, GS_1.best_params_, GS_1.best_score_


param_2 = {'max_depth':[1, 3, 5, 7]}
GS_2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators = 25, learning_rate = 0.1,
                                  max_features = 'auto', subsample = 0.8), 
                       param_grid = param_2, scoring = 'f1_macro',iid = False,cv = 3)
GS_2.fit(x_train, y_train)
GS_2.grid_scores_, GS_2.best_params_, GS_2.best_score_


