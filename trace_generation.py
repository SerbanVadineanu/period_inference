from simso.core import Model
from simso.configuration import Configuration
import random
import numpy as np
import pandas as pd
import sys
import time
from tqdm import tqdm


def StaffordRandFixedSum(n, u, nsets):
    """
    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.

    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value

    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.
    """
    if n < u:
        return None

    #deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = min(int(u), n - 1)
    s = u
    s1 = s - np.arange(k, k - n, -1.)
    s2 = np.arange(k + n, k, -1.) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    x = np.zeros((n, nsets))
    rt = np.random.uniform(size=(n - 1, nsets))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, nsets))  # rand position in simplex
    s = np.repeat(s, nsets)
    j = np.repeat(k + 1, nsets)
    sm = np.repeat(0, nsets)
    pr = np.repeat(1, nsets)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
        sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
        sm = sm + (1.0 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    #iterated in fixed dimension order but needs to be randomised
    #permute x row order within each column
    for i in range(0, nsets):
        x[..., i] = x[np.random.permutation(n), i]

    return x.T.tolist()


def generateLogUniformPeriods(n, minRange, maxRange, basePeriod):
    periods = []
    for i in range(n):
        s = np.log(minRange)
        e = np.log(maxRange + basePeriod)

        # provides a random value with uniform distribution within range [s, e]
        ri = (e - s) * np.random.random_sample() + s
        period = np.floor(np.exp(ri) / basePeriod) * basePeriod
        periods.append(int(period))

    periods = np.sort(periods)
    return periods


def necessary_test(periods, executions):
    pass_test = True
    # Add a necessary test
    indices = np.argsort(periods)
    first_exec = executions[indices[0]]
    first_period = periods[indices[0]]

    for index in indices[1:]:
        if executions[index] > 2 * (first_period - first_exec):
            pass_test = False
            break

    return pass_test


def gen_periods_and_exec(n_tasks=3, total_utilization=0.9, method='automotive', is_preemptive=True):
    # Redo if a rounded execution is 0
    redo = True
    while redo:

        if method == 'automotive':
            periods = np.random.choice([1, 2, 5, 10, 20, 50, 100, 200, 1000],
                                       p=[0.05, 0.03, 0.04, 0.27, 0.27, 0.04, 0.22, 0.02, 0.06],
                                       size=n_tasks)
        elif method == 'loguniform':
            periods = generateLogUniformPeriods(n_tasks, minRange=10, maxRange=1000, basePeriod=10)

        elif method == 'colorado':
            periods = []
            for i in range(n_tasks):
                periods.append(random.randrange(1, 200, 10))

        hyperperiod = np.lcm.reduce(np.array(periods))

        for p in periods:
            # If we have more than 5000 jobs for a task we redo
            if hyperperiod / p > 5000:
                continue

        executions = np.round(StaffordRandFixedSum(n_tasks, total_utilization, 1)[0] * np.array(periods), decimals=2)

        if not is_preemptive:
            test_result = necessary_test(periods, executions)

            # If the test was failed
            if not test_result:
                continue

        if not 0 in executions:
            redo = False

    indices = np.argsort(periods)
    periods = periods[indices]
    executions = executions[indices]

    return periods, executions


# Modified from classification to include zeros for idle time
def create_trace(scheduler="simso.schedulers.RM", n_tasks=3, seed=None, total_utilization=0.9, method='automotive',
                 alpha=0, jitter=0, is_preemptive=True):
    redo = True
    scale = 1
    while redo:
        # Manual configuration:
        configuration = Configuration()
        configuration.cycles_per_ms = 1

        # Replicate the results for more scheduling policies
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Generate periods and executions according to the specified method
        periods, wcets = gen_periods_and_exec(n_tasks, total_utilization, method, is_preemptive)

    #   Debugging
        if method == 'loguniform':
            divider = 1 / 10
        else:
            divider = 1 / 1000
        wcets = wcets / divider
        hyperperiod = np.lcm.reduce(np.array(periods)) / divider
        periods = periods / divider

        wcets = np.round(wcets / scale) * scale

        for i in range(len(wcets)):
            if wcets[i] == 0:
                wcets[i] = scale

        if alpha == 0:

            for i in range(n_tasks):
                configuration.add_task(name="T" + str(i + 1), identifier=i, period=periods[i] / scale,
                                       activation_date=0, wcet=wcets[i] / scale, deadline=periods[i] / scale,
                                       jitter=jitter)

            if jitter == 0:
                configuration.duration = 2 * hyperperiod * configuration.cycles_per_ms / scale  # in seconds
            else:
                configuration.duration = 10 * hyperperiod * configuration.cycles_per_ms / scale  # in seconds

        else:
            configuration.etm = 'ucet'
            ucets = (1 - alpha) * wcets

            for i in range(n_tasks):
                configuration.add_task(name="T" + str(i + 1), identifier=i, period=periods[i] / scale,
                                       activation_date=0, ucet=ucets[i],
                                       wcet=wcets[i] / scale,
                                       deadline=periods[i] / scale,
                                       jitter=jitter)

            configuration.duration = 10 * hyperperiod * configuration.cycles_per_ms / scale  # in seconds

        if configuration.duration < 0:
            continue

        # Add a processor:
        configuration.add_processor(name="CPU 1", identifier=1)

        # Add a scheduler:
        configuration.scheduler_info.clas = scheduler

        # Check the config before trying to run it.
        configuration.check_all()

        # Init a model from the configuration.
        model = Model(configuration)

        # Execute the simulation.
        model.run_model()

        redo = False

    trace = []
    prev_time = 0
    prev_task = None
    for log in model.logs:
        crt_time = log[0]
        info = log[1][0].split("_")
        task = int(info[0].split('T')[1])

        state = info[1].split(' ')

        if 'Preempted!' in state:
            for i in range(1, int((crt_time - prev_time))):
                trace.append(prev_task)
            prev_time = crt_time

        if 'Executing' in state:
            if prev_time != crt_time:
                for i in range(0, int((crt_time - prev_time))):
                    trace.append(0)  # append idle task

            prev_time = crt_time  # reset counting time interval
            prev_task = task
            trace.append(task)

        if 'Terminated.' in state:
            for i in range(1, int((crt_time - prev_time))):
                trace.append(prev_task)
            prev_time = crt_time

    return trace, list(map(int, list(periods)))


def main():

    dataset = str(sys.argv[1])                                          # The type of dataset (automotive or loguniform)
    dataset_size = int(sys.argv[2])                                     # How large will the dataset be
    no_tasks = int(sys.argv[3])                                         # The number of tasks in the trace
    utilization = float(sys.argv[4])                                    # Total utilization of the system
    alpha = float(sys.argv[5])                                          # The fraction of the execution time variation
    jitter = float(sys.argv[6])                                         # The amount of jitter in traces
    is_preemptive = True if str(sys.argv[7]) == 'yes' else False        # Whether the scheduling is preemptive or not

    columns = ['Trace', 'Periods']

    all_traces = []
    all_periods = []

    for _ in tqdm(range(dataset_size)):
        trace, periods = create_trace(n_tasks=no_tasks,
                                      total_utilization=utilization,
                                      method=dataset,
                                      alpha=alpha,
                                      jitter=jitter,
                                      scheduler='simso.schedulers.RM_mono',
                                      is_preemptive=is_preemptive)

        all_traces.append(trace)
        all_periods.append(periods)

    df = pd.DataFrame(list(zip(all_traces, all_periods)), columns=columns)

    path_out = f'{dataset}_{no_tasks}_tasks_{utilization}_utilization'

    if alpha != 0:
        path_out = 'UCET_' + path_out + f'_{alpha}_alpha'
    if jitter != 0:
        path_out = 'JITTER_' + path_out + f'_{jitter}_jitter'
    if alpha == 0 and jitter == 0:
        path_out = 'IDEAL_' + path_out

    if utilization >= 1:
        path_out = 'TARDINESS_' + path_out

    path_out += f'_{int(time.time())}.csv'

    df.to_csv(path_out, index=False)


main()

