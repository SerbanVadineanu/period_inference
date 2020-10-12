# Regression-based period miner
Repository with the solution from the paper "Robust and Accurate Period Inference using Regression-Based Techniques".

## Table of contents
* [General info](#general-info)
* [Trace generator](#trace-generator)
* [Feature extractor](#feature-extractor)
* [Regression algorithms](#regression-algorithms)

## General info
This project contains three major functionalities:
* A **trace generator** which creates a dataset of traces.
* A **feature extractor** which creates a dataset of features from a given set of traces.
* A suite of **tree-based regression algorithms** which can be trained and then used to predict the periods of the tasks within traces. 
	
## Trace generator

Randomly generates a set of traces according to the user's specifications. The result is a csv with two columns: *Trace* and *Periods*. A trace is a list of task identifiers denoted as integers from 1 to the total number of tasks. The IDs are assigned to tasks according to their periods (from lowest to highest). The periods are also stored in a list with their order matching the task IDs.

### Requirements
* Python version: 3.7
* simpy version: 2.3.1
* numpy version: 1.19.2
* pandas version: 1.1.3
* tqdm version: 4.50.2

### Usage
`python trace_generation.py [dataset] [dataset size] [no tasks] [utilization] [alpha] [jitter] [preemptive]`
* *dataset*:
  * **automotive** with periods generated according to [1].
  * **loguniform** with periods generated according to a log-uniform distribution.
* *dataset size*: the number of traces to be generated.
* *no tasks*: the number of tasks within a trace.
* *utilization*: the total uitlization of the system given as a fractional number.
* *alpha*: the amount of execution time variation given as a fractional number.
* *jitter*: the amount with release jitter w.r.t. the period of th task given as a fractional number.
* *preemptive*: whether or not the execution model is preemptive.
  * **yes** for preemptive execution.
  * **no** for non-preemptive execution.

### Output format

|Trace|Periods|
|---|---|
|'[1, 1, 1, 2, 3, 3, 0, 0, 1, 1, 1, 2, 3, 3]'|'[8, 9, 10]'|

Both the trace and the task periods are saved as strings representing lists such that they can be easily parsed when read by the [Feature extractor](#feature-extractor).

### Example
` python trace_generation.py automotive 300 4 0.7 0 0 yes `

The script will generate a csv file containing 300 traces, each trace with 4 tasks and a total utilization of 70%. Also, there is no execution time variation, no release jitter and the execution is preemptive.

The output file will be named `IDEAL_automotive_4_tasks_0.7_utilization_1602426197.csv` meaning that the execution is without uncertainties (IDEAL), with other distinguishing parameters mentioned in the name and ending with the timestamp.

## Feature extractor

Receives as input a set of traces under the format specified in [Trace generator](#trace-generator) and creates a set of features which can be used either for training or for prediction.

### Requirements
* Python version: 3.7
* numpy version: 1.19.2
* pandas version: 1.1.3
* tqdm version: 4.50.2

### Usage
` python feature_extraction.py [path to dataset] [no features] [training]`
* *path to dataset*: the path to the data set of traces.
* *no features (recommended 20)*: the number of features to be extracted from the two signal processing techniques, meaning that the number of resulting features will be double the specified value.
* *training*: whether or not the data set of traces will be used for training or for prediction.
  * **yes** for creating a training set of features, meaning that this set will also include the true period for each task. Thus, the list of periods **must** be specified in the data set of traces.
  * **no** for creating a set of features for prediction. Here, the true periods may be omitted.

The script will generate a csv file containing the features extracted from the projection of each task from each trace.
Also, the resulted file will contain the lower bound and the upper bound required to use the space prunning method.

### Output format

|Top1_periodogram|Top2_periodogram|...|Top20_periodogram|Top1_autocorrelation|...|Top20_autocorrelation|Lower_bound|Upper_bound|True_period|
|---|---|---|---|---|---|---|---|---|---|
|1000.0|500.0|...|43.0|2000.0|...|20000.0|392.0|2000.0|1000.0|

### Example
`python feature_extraction.py python feature_extraction.py /Users/myuser/period_inference/IDEAL_automotive_4_tasks_0.7_utilization_1602426197.csv 20 yes`

The result will be a file called `features_training_1602426752.csv` denoting that this data set incorporates the true periods in it.
In case the the *training* parameter would be **no**, then the result would be `features_testing_1602426752.csv`.

## Regression algorithms

Receives as input a set of features and generates either a trained model or a set of predictions.

### Requirements
* R version: 4.0.1
* rJava library version: 0.9-12
* CARET library version: 6.0-86
* Cubist library version: 0.2.3
* extraTrees library version: 1.0.5
* bartMachine library version: 1.2.4.2

### Usage
` Rscript run_algorithm.r [algorithm] [path to dataset] [path to model]`

* *algorithm*: the regression algorithm to be used.
  * **cubist** [2]
  * **extraTrees** [3]
  * **gbm** [4]
  * **bartMachine** [5]
* *path to dataset*: the path to the data set containing the features to be used either for training or for prediction
* *path to model*: the path to the trained regression model. In case of training this argument **should not** be introduced.

When training, the result will be a rds file with the following naming format: `model_[algorithm].rds`.

When predicting, the result will be a csv file containing the predicted period for each task represented by its set of features. 
The csv will include 3 columns:
* *RPM*: the predictions coming directly from the regression algorithms
* *RPMPA*: the predictions from the period adjustment step
* *SPM*: the predictions from the space prunning method

### Output format

|RPM|RPMPA|SPM|
|---|---|---|
|999.99|1000|1000|

### Example

* Training: `Rscript run_algorithm.r cubist /Users/myuser/period_inference/features_training_1602426752.csv`
* Predicting: `Rscript run_algorithm.r cubist /Users/myuser/period_inference/features_testing_1602426752.csv /Users/myuser/period_inference/model_cubist.rds`

## Bibliography
[1] Kramer, S., Ziegenbein, D., & Hamann, A. (2015, July). Real world automotive benchmarks for free. In 6th International Workshop on Analysis Tools and Methodologies for Embedded and Real-time Systems (WATERS).

[2] Max Kuhn and Ross Quinlan (2020). Cubist: Rule- And Instance-Based Regression Modeling. R package version 0.2.3. https://CRAN.R-project.org/package=Cubist.

[3] Simm J, de Abril I, Sugiyama M (2014). _Tree-Based Ensemble Multi-Task Learning Method for Classification and Regression_, volume 97 number 6. <URL: http://CRAN.R-project.org/package=extraTrees.

[4] Brandon Greenwell, Bradley Boehmke, Jay Cunningham and GBM Developers (2019). gbm: Generalized Boosted Regression Models. R package version 2.1.5. https://CRAN.R-project.org/package=gbm.

[5] Adam Kapelner, Justin Bleich (2016). bartMachine: Machine Learning with Bayesian Additive Regression Trees. Journal of Statistical Software, 70(4), 1-40. doi:10.18637/jss.v070.i04. https://CRAN.R-project.org/package=bartMachine.

