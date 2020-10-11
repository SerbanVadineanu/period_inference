
options(java.parameters = '-xmx60g')

library(bartMachine)
library(Cubist)
library(mlbench)
library(dplyr)
library(pryr)
library(caret)
library(extraTrees)

combined_predict_limits <- function(df_features, predictions, lower_bound, upper_bound) {
  y_pa <- double(length(predictions))
  y_spm <- double(length(predictions))

  for (row in 1:nrow(df_features)) {

    features <- unique(as.integer(df_features[row, ]))

    # Period adjustment method (RPMPA)
    y_pa[row] = features[which.min(abs(features - predictions[row]))]

    # Space prunning method (SPM)
    selected = c()
    for (f in features) {
      if(f >= lower_bound[row] && f <= upper_bound[row]) {
        selected <- c(selected, f)
      }
    }
    if(length(selected) == 0){
      y_spm[row] <- upper_bound[row]
    } else {
      y_spm[row] = selected[which.min(abs(selected - predictions[row]))]
    }
  }

  return(list(y_pa=y_pa, y_spm=y_spm))

}

run_algorithm <- function (algorithm, dataset_path, model_path) {
  # Read the data
  data <- read.csv(dataset_path)
  X <- select(data,
              Top1_periodogram, Top2_periodogram, Top3_periodogram,
              Top1_autocorrelation, Top2_autocorrelation, Top3_autocorrelation)

  # If no model path is given then we assume training
  if (is.null(model_path)) {
    # Separate features from labels
    y <- select(data, True_period)

    # For hyperparameter tuning
    control <- trainControl(
      method = "boot",
      p = 0.75,
      number = 1,
      search = 'random',
      classProbs = T)

    if(identical(algorithm, "gbm")){
      model <- train(x = X,
                     y = y[,],
                     method = algorithm,
                     trControl = control,
                     tuneLength = 1,
                     verbose = FALSE)
    } else if (identical(algorithm, "cubist")) {
      model <- Cubist::cubist(x = X, y = y[,])
    } else if (identical(algorithm, "extraTrees")) {
      model <- extraTrees::extraTrees(x = X, y = y[,])
      extraTrees::prepareForSave(model)
    } else if (identical(algorithm, "bartMachine")) {
      model <- bartMachine::bartMachine(X = X, y = y[,],
                                        verbose = FALSE,
                                        serialize = TRUE)
    }

    # Save the trained model
    saveRDS(model, paste0("./model_", algorithm, ".rds"))

  } else {   # Otherwise we assume testing
    
    model <- readRDS(model_path)
    
    if (identical(algorithm, "bartMachine")){
      y_pred <- predict(model, new_data = X)
    } else {
      y_pred <- predict(model, newdata = X)
    }

    # Select first 40 columns corresponding to the 20 features from periodogram and autocorrelation
    data %>% select(1:40)
    lower_bound <- select(data, Lower_bound)[,]
    upper_bound <- select(data, Upper_bound)[,]
    combined_results <- combined_predict_limits(data, y_pred, lower_bound, upper_bound)
    # Results from the period adjustment step
    y_pa <- combined_results$y_pa
    # Results from the space prunning method
    y_spm <- combined_results$y_spm

    columns <- c('RPM', 'RPMPA', 'SPM')
    predictions_df <- data.frame(y_pred, y_pa, y_spm)
    colnames(predictions_df) <- columns

    write.csv(predictions_df, paste0('predictions_', algorithm, '.csv'),
              row.names = FALSE)
  }
}

# -------- main --------
args = commandArgs(trailingOnly = TRUE)

algorithm = args[1]
dataset_path = args[2]
model_path = NULL

if (length(args) == 3){
  model_path = args[3]
}

run_algorithm(algorithm, dataset_path, model_path)
