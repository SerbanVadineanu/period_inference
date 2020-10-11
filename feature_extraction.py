import numpy as np
import pandas as pd
from scipy import signal
import sys
import time
import math
from ast import literal_eval
from tqdm import tqdm


# Creates a projection of the trace w.r.t. a certain task
def project_trace(trace, task, is_ternary=False):
    result_trace = []
    # Variable to check if there is at least a 1 in the trace (for the case of missed deadlines)
    not_one = True

    if is_ternary:
        for symbol in trace:
            if symbol == task:
                not_one = False
                result_trace.append(1)
            elif symbol == 0:
                result_trace.append(-1)  # We denote the idle task by -1
            else:
                result_trace.append(0)

    else:
        for symbol in trace:
            if symbol == task:
                not_one = False
                result_trace.append(1)
            else:
                result_trace.append(0)

    # If no 1 is found in the trace
    if not_one:
        return None

    return result_trace


# Top n peaks of periodogram and top n peaks of autocorrelation
def features_from_trace(trace, no_features=3):
    features = []

    if trace is None:
        return list(np.zeros(2*no_features))

    # Periodogram features
    periodogram = signal.periodogram(trace)

    # Check for zero values
    sorted_freq = periodogram[0][np.argsort(-periodogram[1])]
    top_periodogram = 1 / sorted_freq[:no_features]

    i, = np.where(top_periodogram == math.inf)
    if i.size > 0:
        top_periodogram = np.delete(top_periodogram, i)
        top_periodogram = np.append(top_periodogram, 1 / sorted_freq[no_features])

    top_periodogram_approx = [int(p) for p in top_periodogram]

    for p in top_periodogram_approx:
        features.append(p)

    # Autocorrelation features
    autocorrelation = signal.correlate(trace, trace, mode='full')[len(trace) - 1:]

    peaks = signal.find_peaks(autocorrelation)

    try:
        highest_peak = peaks[0][np.argsort(-autocorrelation[peaks[0]])[0]]
        # Check for fewer peaks than features
        if len(peaks[0]) < no_features:
            top_autocorr = peaks[0][np.argsort(-autocorrelation[peaks[0]])][:len(peaks[0])]
            for i in range(no_features - len(peaks[0])):
                top_autocorr = np.append(top_autocorr, highest_peak)
        else:
            top_autocorr = peaks[0][np.argsort(-autocorrelation[peaks[0]])][:no_features]

        for a in top_autocorr:
            features.append(a)

    except Exception:
        features = list(np.zeros(2 * no_features))
    finally:
        return features


# Extract the lower bound and the upper bound from trace
def get_bounds(trace):
    first_idle = True
    prev_idle = 0
    fin = 0
    recent_idle = 0
    delta = 0
    lower_bound = 0
    upper_bound = math.inf
    task_ran_after_idle = None

    for k in range(1, len(trace)):
        if trace[k] != 1:  # If we find non-running time for our task
            if trace[k-1] == 1:    # And previously the task was running
                delta = 1
            else:
                delta += 1

        if trace[k] == 1:   # If the current task is running
            fin = k
            task_ran_after_idle = True
            if trace[k-1] != 1:     # And it was not running before
                lower_bound = max(delta/2, lower_bound)

        if trace[k] != -1 and trace[k-1] == -1:     # If there is a transition from idle to running
            if first_idle:  # If it is the first idle time to be found
                recent_idle = k
                first_idle = False
                task_ran_after_idle = False
            elif task_ran_after_idle:
                upper_bound = min(fin - prev_idle, upper_bound)
                prev_idle = recent_idle
                recent_idle = k
                task_ran_after_idle = False

    return lower_bound, upper_bound


# Assumption: tasks indices are integers from 1 to n and are ordered based on the period (task i -> periods[i-1])
def main():

    path_dataset = str(sys.argv[1])                             # Path to the dataset with traces
    no_features = int(sys.argv[2])                              # Number of features to be extracted
    is_training = True if str(sys.argv[3]) == 'yes' else False  # Are the features used for training or testing

    columns_1 = []

    for i in range(no_features):
        columns_1.append(f'Top{i+1}_periodogram')
    for i in range(no_features):
        columns_1.append(f'Top{i+1}_autocorrelation')

    columns_2 = ['Lower_bound', 'Upper_bound']
    columns = columns_1 + columns_2

    if is_training:
        columns.append('True_period')

    dataset = pd.read_csv(path_dataset)

    df = pd.DataFrame(columns=columns)
    for index, row in tqdm(dataset.iterrows()):

        trace = literal_eval(row['Trace'])
        if is_training:
            periods = literal_eval(row['Periods'])
        tasks = np.sort(np.unique(trace))

        for i in range(len(tasks) - 1):     # -1 because we do not include 0, denoted for idle task
            projected_trace_binary = project_trace(trace=trace, task=i + 1)
            projected_trace_ternary = project_trace(trace=trace, task=i + 1, is_ternary=True)

            LB, UB = get_bounds(projected_trace_ternary)

            line = features_from_trace(projected_trace_binary, no_features=no_features)
            line.append(LB)
            line.append(UB)

            if is_training:
                line.append(periods[i])

            df.loc[len(df)] = line

    path_out = f'features'

    if is_training:
        path_out += '_training_'
    else:
        path_out += '_testing_'

    path_out += f'{int(time.time())}.csv'
    df.to_csv(path_out, index=False)


main()