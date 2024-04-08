library(tidyverse)
library(caret)
library(haven)
library(parallel)
library(doParallel)

library(tidyverse)
library(caret)
library(haven)
library(parallel)
library(doParallel)

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
local_cluster <- makeCluster(2)
registerDoParallel(local_cluster)

m1_time <- system.time(model1 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "ranger",
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

m1_time


stopCluster(local_cluster)
registerDoSEQ()

