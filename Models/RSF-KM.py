import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import KFold
from sksurv.util import Surv
import time

def internalCV_RSF(training, numFolds):
    start_time = time.time()
    kf = KFold(n_splits=numFolds)
    resultsMatrix = np.zeros((numFolds, 3))

    for i, (train_index, test_index) in enumerate(kf.split(training)):
        trainingFold = training.iloc[train_index]
        testingFold = training.iloc[test_index]

        resultVec = np.array([])
        for ntree in [50]:
            for nodesize in [2, 3, 5]:
                print(f"Training RSF with ntree={ntree} and nodesize={nodesize}")
                rsfMod = RandomSurvivalForest(n_estimators=ntree, min_samples_split=nodesize)
                rsfMod.fit(trainingFold.drop(['time', 'delta'], axis=1), Surv.from_dataframe('delta', 'time', trainingFold))
                error = 1 - rsfMod.score(testingFold.drop(['time', 'delta'], axis=1), Surv.from_dataframe('delta', 'time', testingFold))
                resultVec = np.append(resultVec, error)

        resultsMatrix[i, :] = resultVec

    meanResults = np.mean(resultsMatrix, axis=0)
    bestRes = np.argmin(meanResults)
    bestNtree = [50][bestRes // 3]
    bestSize = [2, 3, 5][3 if bestRes % 3 == 0 else bestRes % 3]
    elapsed_time = time.time() - start_time
    print(f"CV Completed in {elapsed_time:.2f} seconds.")
    print(f"Best Parameters - ntree: {bestNtree}, nodesize: {bestSize}")
    return {'ntree': bestNtree, 'nodesize': bestSize}

def RSF(training, testing, params, numFolds=5):
    ntree = 100
    nodesize = 5

    # Check if 'delta' column is binary
    if len(training['delta'].unique()) != 2 or set(training['delta'].unique()) != {0, 1}:
        raise ValueError("Event indicator ('delta') must be binary (0 or 1).")

    rsfMod = RandomSurvivalForest(n_estimators=ntree, min_samples_split=nodesize)
    rsfMod.fit(training.drop(['time', 'delta'], axis=1), Surv.from_dataframe('delta', 'time', training))

    survivalCurves = rsfMod.predict_survival_function(testing.drop(['time', 'delta'], axis=1))
    survivalCurvesTrain = rsfMod.predict_survival_function(training.drop(['time', 'delta'], axis=1))
    trainingTimes = survivalCurves[0].x  # Assuming the first survival curve has the times of interest

    print(survivalCurves)
    # If 0 wasn't included in the timepoints, manually add it with a survival probability of 1
    if 0 in trainingTimes:
        times = trainingTimes
        testProbabilities = pd.DataFrame({f'Individual_{i}': survivalCurves[i].y for i in range(len(survivalCurves))}, index=times)
        trainProbabilities = pd.DataFrame({f'Individual_{i}': survivalCurvesTrain[i].y for i in range(len(survivalCurvesTrain))}, index=times)
    else:
        times = [0] + list(trainingTimes)
        # Create a DataFrame for test survival probabilities
        testProbabilities = pd.DataFrame({f'Individual_{i}': [1] + list(survivalCurves[i].y) for i in range(len(survivalCurves))}, index=times)
        # Create a DataFrame for train survival probabilities
        trainProbabilities = pd.DataFrame({f'Individual_{i}': [1] + list(survivalCurvesTrain[i].y) for i in range(len(survivalCurvesTrain))}, index=times)

    # # Create the curvesToReturn structure
    # curvesToReturn = {'TestCurves': testProbabilities, 'TrainingCurves': trainProbabilities}

    # timesAndCensTest = pd.DataFrame({'time': testing['time'], 'delta': testing['delta']})
    # timesAndCensTrain = pd.DataFrame({'time': training['time'], 'delta': training['delta']})

    # return {'TestCurves': curvesToReturn, 'TestData': timesAndCensTest,
    #         'TrainData': timesAndCensTrain, 'TrainCurves': curvesToReturn['TrainingCurves']}

    return {
        'TestCurves': testProbabilities,
        'TestData': testing[['time', 'delta']],
        'TrainData': training[['time', 'delta']],
        'TrainCurves': trainProbabilities
    }

