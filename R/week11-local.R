#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)

# Data Import and Cleaning
gss_import_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(MOSTHRS)) %>%
  select(-HRS1, -HRS2) 

gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < .75 * nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric))


# Visualization
ggplot(gss_tbl, aes(x=MOSTHRS)) +
  geom_histogram()
  

# Analysis
holdout_indices <- createDataPartition(gss_tbl$MOSTHRS,
                                       p = .25,
                                       list = T)$Resample1

training_tbl <- gss_tbl[holdout_indices,]
test_tbl <- gss_tbl[-holdout_indices,]

training_folds <- createFolds(training_tbl$MOSTHRS)

model1 <- train(
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

cv_m1 <- model1$results$Rsquared
holdout_m1 <- cor(
  predict(model1, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2




model2 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
)

cv_m2 <- max(model2$results$Rsquared)
holdout_m2 <- cor(
  predict(model2, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


model3 <- train(
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

cv_m3 <- max(model3$results$Rsquared)
holdout_m3 <- cor(
  predict(model3, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2


model4 <- train(
  MOSTHRS ~ .,
  training_tbl,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = c("center","scale", "nzv", "medianImpute"),
  trControl = trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = TRUE, 
                           indexOut = training_folds)
)

cv_m4 <- max(model4$results$Rsquared)
holdout_m4 <- cor(
  predict(model4, test_tbl, na.action = na.pass),
  test_tbl$MOSTHRS
) ^ 2
holdout_m4

# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format = "f", digits = 2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table1_tbl <- tibble(
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
table1_tbl
