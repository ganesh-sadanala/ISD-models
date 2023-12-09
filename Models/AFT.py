import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

def AFT(training, testing, AFTDistribution):
    aft_model = None

    if AFTDistribution == 'weibull':
        aft_model = WeibullAFTFitter()
    elif AFTDistribution == 'lognormal':
        aft_model = LogNormalAFTFitter()
    elif AFTDistribution == 'loglogistic':
        aft_model = LogLogisticAFTFitter()
    else:
        print("Unsupported AFT distribution:", AFTDistribution)
        return None

    aft_model.fit(training, duration_col='time', event_col='delta')

    # times_to_predict = np.sort(testing['time'].unique())
    # Generate a continuous range of time points for prediction
    min_time = min(training['time'].min(), testing['time'].min())
    max_time = max(training['time'].max(), testing['time'].max())
    times_to_predict = np.linspace(min_time, max_time, 20000)
    if 0 not in times_to_predict:
        times_to_predict = np.insert(times_to_predict, 0, 0)

    test_curves = aft_model.predict_survival_function(testing, times=times_to_predict)
    train_curves = aft_model.predict_survival_function(training, times=times_to_predict)

    return {
        'TestCurves': test_curves,
        'TestData': testing[['time', 'delta']],
        'TrainData': training[['time', 'delta']],
        'TrainCurves': train_curves
    }

# The following functions are for survival function calculation, similar to the R code's `survfunc`.
def survfunc(aft_model, t, newdata, name="t"):
    if name == "t":
        t_col = newdata[name]
        newdata = newdata.drop(name, axis=1)
    else:
        t_col = newdata["t"]

    try:
        pdf = aft_model.predict_density(newdata, t_col)
        cdf = aft_model.predict_survival_function(newdata, t_col)
        haz = pdf / (1 - cdf)

        newdata["pdf"] = pdf
        newdata["cdf"] = cdf
        newdata["haz"] = haz

        sur = 1 - cdf
        newdata[name] = t_col
        newdata["sur"] = sur
        return newdata
    except Exception as e:
        print(e)
        return None
