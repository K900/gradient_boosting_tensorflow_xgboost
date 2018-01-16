#!/usr/bin/env python2

"""
Train a gradient boosting classifier on the airline dataset using
CatBoost's Python API.
"""

import argparse

import numpy as np
import pandas as pd
from IPython import embed
import pickle
import scipy
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot as plt
import seaborn as sns

FLAGS = None


def train_and_predict(X_train, y_train, X_test, y_test, **kwargs):
    """Run training and evaluation using CatBoost."""

    bst = CatBoostClassifier(
        depth=FLAGS.depth,
        learning_rate=FLAGS.learning_rate,
        iterations=FLAGS.num_trees,
        thread_count=4,
        random_seed=42,
    )
    bst.fit(X_train, y_train, cat_features=[3, 4, 5])
    # pickle.dump(bst, open('catboost.pickle', 'wb'))
    y_pred = bst.predict_proba(X_test)[:, 1]

    # Save predictions
    np.save(
        'outputs/pred_cat_t{:03d}_d{:02d}.npy'.format(FLAGS.num_trees, FLAGS.depth),
        y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_trees",
        type=int,
        default=50,
        help="Number of trees to grow before stopping.")
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Maximum depth of weak learners.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (shrinkage weight) with which each new tree is added.")

    FLAGS = parser.parse_args()

    data = np.load('airlines_data.npz')
    train_and_predict(**data)
