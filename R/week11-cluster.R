#Script Settings and Resources
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(parallel)
library(doParallel)

# Data Import and Cleaning
gss_import_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(MOSTHRS)) %>%
  select(-HRS1, -HRS2) 

gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < .75 * nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric))


# Analysis


holdout_indices <- createDataPartition(gss_tbl$MOSTHRS,
                                       p = .25,
                                       list = T)$Resample1

training_tbl <- gss_tbl[holdout_indices,]
test_tbl <- gss_tbl[-holdout_indices,]

training_folds <- createFolds(training_tbl$MOSTHRS)

m1_time <- system.time(model1 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
)
)
cv_m1 <- model1$results$Rsquared
holdout_m1 <- cor(
  predict(model1, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2



m2_time <- system.time(model2 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cv_m2 <- max(model2$results$Rsquared)
holdout_m2 <- cor(
  predict(model2, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


m3_time <- system.time(model3 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cv_m3 <- max(model3$results$Rsquared)
holdout_m3 <- cor(
  predict(model3, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


m4_time <- system.time(model4 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cv_m4 <- max(model4$results$Rsquared)
holdout_m4 <- cor(
  predict(model4, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2



local_cluster <- makeCluster(8)
registerDoParallel(local_cluster)
m1_parallel_time <- system.time(model1 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "lm",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
)
)
cvp_m1 <- model1$results$Rsquared
holdout_m1 <- cor(
  predict(model1, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2




m2_parallel_time <- system.time(model2 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cvp_m2 <- max(model2$results$Rsquared)
holdout_m2 <- cor(
  predict(model2, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


m3_parallel_time <- system.time(model3 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cvp_m3 <- max(model3$results$Rsquared)
holdout_m3 <- cor(
  predict(model3, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


m4_parallel_time <- system.time(model4 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
))

cvp_m4 <- max(model4$results$Rsquared)
holdout_m4 <- cor(
  predict(model4, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


stopCluster(local_cluster)
registerDoSEQ()


# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format = "f", digits = 2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table3 <- tibble(
  algo <- c("regression", "elastic net", "random forests", "xgboost"),
  cv_rqs = c(make_it_pretty(cv_m1), 
             make_it_pretty(cv_m2), 
             make_it_pretty(cv_m3), 
             make_it_pretty(cv_m4)),
  ho_rqs = c(
    make_it_pretty(holdout_m1),
    make_it_pretty(holdout_m2),
    make_it_pretty(holdout_m3),
    make_it_pretty(holdout_m4)
  )
)
table3
dotplot(resamples(list(model1, model2, model3, model4)), metric = "Rsquared")


table4 <- tibble(
  algo = c("regression", "elastic net", "random forests", "xgboost"),
  supercomputer = c(m1_time[3], m2_time[3], m3_time[3], m4_time[3]),
  supercomputer_30 = c(m1_parallel_time[3], m2_parallel_time[3], m3_parallel_time[3], m4_parallel_time[3])
)
table4

write.table(table3, "table3.csv", sep = ",")
write.table(table4, "table4.csv", sep = ",")


## Questions
## 1. Interestingly, and though this seems like it may be due to an error on my part, all of my models took longer to run
## when parallelizing. Perhaps this is do to the amount of resources I requested, or that for some reason I could not get access
## to mangi and so had to use mesabi. The code I am using is identical to what was run locally, in which parallelizing was faster,
## so it seems like an odd outcome. My best guess is that the initial system processing that goes into setting up our local cluster
## takes longer on msi, at least the way I set it up, so if the job we are having it complete is comparatively short any gains are washed out. 
## So maybe MSI is designed to optimize massive workloads on large datasets at the expense of gains on smaller ones?

## 2. Initially I ran into a lot of issues when testing core numbers in the compute node. I tried using higher numbers >15 cores
## and it resulted in memory failure and the killing of my qeued R script. So I moved it back down to 8 cores, which worked on my local machine,
## and that seemed to work fine. My guess is that in most cases more cores = faster processing (assuming the code is sufficiently parallizable),
## but again for whatever reason in this instance it was not. It also seems likely this relationship wouldn't be linear,
## and would be one with diminishing returns, because the more cores used the less work being dedicated to each, so less time available to gain per core.

## 3. I would pick random forest. While it takes significantly longer to run than lm and glm (122 > 12 seconds), the holdout R^2 value is significantly higher (.61 > .36), so that 
## extra time feels worthwile. If I was tight on time I would choose the glm, as it significantly outperforms the lm, and performs
## comparably to our xgboost in far less time.


