import numpy as np
from scipy.integrate import quad
from lifelines import KaplanMeierFitter

def predict_probability_from_curve(survival_curve, predicted_times, time_to_predict):
    max_time = np.max(predicted_times)
    spline = np.poly1d(np.polyfit(predicted_times, survival_curve, 3))

    def spline_with_linear(time):
        return np.where(time < max_time, spline(time), 1 + (time - max_time) * (1 - spline(max_time)) / (0 - max_time))

    predicted_probabilities = np.where(time_to_predict > max_time, np.maximum(1 + time_to_predict * (1 - spline(max_time)) / (0 - max_time), 0), spline(time_to_predict))

    return predicted_probabilities

def predict_mean_survival_time_spline(survival_curve, predicted_times):
    max_time = np.max(predicted_times)
    spline = np.poly1d(np.polyfit(predicted_times, survival_curve, 3))
    slope = (1 - spline(max_time)) / (0 - max_time)
    zero_probability_time = min(predicted_times[survival_curve == 0], max_time + (0 - spline(max_time)) / slope)

    def spline_with_linear(time):
        return np.where(time < max_time, spline(time), 1 + time * slope)

    area, _ = quad(spline_with_linear, 0, zero_probability_time, points=1000)

    return area

def predict_median_survival_time_spline(survival_curve, predicted_times):
    survival_curve = np.asarray(survival_curve)
    min_prob = np.min(np.poly1d(np.polyfit(predicted_times, survival_curve, 3))(predicted_times))
    if min_prob < 0.5:
        maximum_smaller_than_median = predicted_times[np.min(np.where(survival_curve < 0.5))]
        minimum_greater_than_median = predicted_times[np.max(np.where(survival_curve > 0.5))]
        spline_inv = np.poly1d(np.polyfit(np.poly1d(np.polyfit(predicted_times, survival_curve, 3))(np.linspace(minimum_greater_than_median, maximum_smaller_than_median, num=1000)),
                                         np.linspace(minimum_greater_than_median, maximum_smaller_than_median, num=1000), 3))
        median_probability_time = spline_inv(0.5)
    else:
        max_time = np.max(predicted_times)
        slope = (1 - spline(max_time)) / (0 - max_time)
        median_probability_time = max_time + (0.5 - spline(max_time)) / slope

    return median_probability_time

def l1_measure(surv_mod, l1_type="Margin", log_scale=False, method="Median"):
    # if surv_mod is None or np.isnan(surv_mod[0][0]):
    #     return None

    predicted_times = surv_mod[0]["time"]
    survival_curves = surv_mod[0].survival_probabilities
    true_death_times = surv_mod[1]["time"]
    censor_status = surv_mod[1]["delta"]
    censor_times = true_death_times[~censor_status.astype(bool)]
    training_death_times = surv_mod[2]["time"]
    training_censor_status = surv_mod[2]["delta"]

    def predict_method(id):
        if method == "Mean":
            return predict_mean_survival_time_spline([row[id] for row in survival_curves], predicted_times)
        elif method == "Median":
            return predict_median_survival_time_spline([row[id] for row in survival_curves], predicted_times)
        else:
            return None

    print(np.where(1-censor_status)[0])
    average_uncensored = np.array([predict_method(index) for index in np.where(censor_status)[0]])
    average_censored = np.array([predict_method(index) for index in np.where(1-censor_status)[0]])

    # Sometimes the mean is infinite or extremely large. Enforce that a model cannot predict a value higher
    # than the end of the KM curve with the linear extension.
    km_curve = KaplanMeierFitter().fit(training_death_times, event_observed=training_censor_status)
    km_linear_zero = -1 / ((1 - min(km_curve.survival_function_.iloc[-1])) / (0 - max(km_curve.timeline)))

    # If every patient is censored, choose the last time point to be the maximum time.
    if np.isinf(km_linear_zero):
        km_linear_zero = np.max(km_curve.time)

    average_uncensored = np.minimum(average_uncensored, km_linear_zero)
    average_censored = np.minimum(average_censored, km_linear_zero)

    uncensored_piece = np.sum(np.abs(true_death_times[censor_status.astype(bool)] - average_uncensored))

    if not log_scale:
        uncensored_piece = np.sum(np.abs(true_death_times[censor_status.astype(bool)] - average_uncensored))
    else:
        uncensored_piece = np.sum(np.abs(np.log(true_death_times[censor_status.astype(bool)]) - np.log(average_uncensored)))

    if l1_type == "Uncensored":
        l1_measure = (1 / np.sum(censor_status)) * uncensored_piece
    elif l1_type == "Hinge":
        hinge_piece = np.where(~log_scale,
                               censor_times - average_censored,
                               np.log(censor_times) - np.log(average_censored))
        hinge_piece_corrected = np.where(hinge_piece >= 0, hinge_piece, 0)
        l1_measure = (1 / len(censor_status)) * (uncensored_piece + np.sum(hinge_piece_corrected))
    elif l1_type == "Margin":
        def km_linear_predict(time, km_curve):
          prediction = 1 - km_curve.predict(time).squeeze()
          slope = (1 - min(km_curve.survival_function_.iloc[-1])) / (0 - max(km_curve.timeline))
          predicted_probabilities = np.maximum(1 + time * slope, prediction)
          return predicted_probabilities

        best_guess = np.array([time + np.trapz(km_linear_predict(time, km_curve), dx=1e-2) / km_linear_predict(time, km_curve)
                              for time in censor_times])
        best_guess[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]
        weights = 1 - km_linear_predict(censor_times, km_curve)

        # Inside l1_measure function
        if not log_scale:

            if weights.size == 0:
                margin_piece = 0
            else:
                margin_piece = np.sum(weights * np.abs(best_guess - average_censored))
        else:
            # Similar print statements if needed
            if weights.size == 0:
                margin_piece = 0
            else:
                margin_piece = np.sum(weights * np.abs(np.log(best_guess) - np.log(average_censored)))

        # Calculate L1 measure
        l1_measure = (1 / np.sum(censor_status)) * uncensored_piece + (1 / np.sum(weights)) * margin_piece
    else:
        l1_measure = None

    return l1_measure
