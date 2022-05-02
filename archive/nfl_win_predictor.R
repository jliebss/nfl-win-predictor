# ---- Load Packages ----
library(conflicted)
library(nflfastR)
library(tidyverse)
library(MASS)
library(e1071)
library(caret)
library(class)
library(rpart)
library(iml)


# ---- Useful Code ----
# Time Code
# ptm <- proc.time()
# proc.time() - ptm

# Train/Test Split
# set.seed(123)
# smp_size  <- floor(0.75 * nrow(box_score))
# train_ind <- sample(seq_len(nrow(box_score)), size = smp_size)
# train <- box_score[train_ind,  -c(1, 2, ncol(box_score)-1)]
# test  <- box_score[-train_ind, -c(1, 2, ncol(box_score)-1)]

# Remove NAs
# index <- apply(df, 2, function(x) any(is.na(x)))
# colnames(df[index])
# df <- df[complete.cases(df$feature), ]
# df %>% drop_na()

# Select All Columns Except
# features <- as.data.frame(box_score[, !names(box_score) %in% c("game_id", "posteam", "winner_name", "winner")])


# ---- Load Data ----
raw_data <- nflfastR::load_pbp(2017:2021)
raw_data <- clean_pbp(raw_data)
raw_data <- raw_data[complete.cases(raw_data$posteam), ]

schedule <- fast_scraper_schedules(2017:2021) %>% 
  mutate(winner_name = case_when(home_result > 0 ~ home_team,
                                 home_result < 0 ~ away_team,
                                 TRUE            ~ "TIE"))


# ---- Prepare and Partition Data ----

box_score <- raw_data %>% 
  group_by(game_id, posteam) %>% 
  summarize_at(vars(yards_gained, air_yards, yards_after_catch, epa, air_epa, yac_epa, comp_air_epa, comp_yac_epa, wpa, vegas_wpa, air_wpa, yac_wpa, comp_air_epa, comp_yac_epa, punt_blocked, first_down_rush, first_down_pass, first_down_penalty,
                    third_down_converted, third_down_failed, fourth_down_converted, fourth_down_failed, incomplete_pass, touchback, interception, punt_inside_twenty, punt_in_endzone, punt_out_of_bounds, punt_downed, punt_fair_catch,
                    kickoff_inside_twenty, kick_distance, kickoff_in_endzone, kickoff_out_of_bounds, kickoff_fair_catch, kickoff_attempt, kickoff_downed, fumble_forced, fumble_not_forced, fumble_out_of_bounds, fumble_lost, fumble, solo_tackle,
                    safety, penalty, tackled_for_loss, own_kickoff_recovery, qb_hit, rush_attempt, pass_attempt, sack, touchdown, pass_touchdown, rush_touchdown, return_touchdown, extra_point_attempt, two_point_attempt,
                    field_goal_attempt, punt_attempt, complete_pass, assist_tackle, lateral_reception, lateral_rush, lateral_return, lateral_recovery, passing_yards, receiving_yards, rushing_yards, lateral_receiving_yards, lateral_rushing_yards,
                    return_yards, penalty_yards, special_teams_play, aborted_play, success, pass, rush, first_down, special, play, out_of_bounds),
               list(name = sum),
               na.rm = TRUE) %>% 
  merge(schedule[, c("game_id", "winner_name")], by = "game_id", all.x = TRUE) %>% 
  mutate(winner = factor(case_when(posteam == winner_name ~ 1,
                                   TRUE                   ~ 0)))
# rm(raw_data)
# nflreadr::.clear_cache()

set.seed(123)
smp_size  <- floor(0.75 * nrow(box_score))
train_ind <- sample(seq_len(nrow(box_score)), size = smp_size)

train <- box_score[train_ind,  -c(1, 2, ncol(box_score)-1)]
test  <- box_score[-train_ind, -c(1, 2, ncol(box_score)-1)]

train_scaled <- train
test_scaled  <- test
train_scaled[, -ncol(train_scaled)] <- scale(train_scaled[, -ncol(train_scaled)])
test_scaled[, -ncol(test_scaled)]   <- scale(test_scaled[, -ncol(test_scaled)])


# ---- Logistic Regression Models ----
# Base case
logit_fit   <- glm(winner ~ .,
                 data   = train,
                 family = binomial)
logit_probs <- predict(logit_fit,
                     newdata = test,
                     type    = "response")
logit_pred  <- ifelse(logit_probs > 0.5, 1, 0)

logit_table <- table(test$winner, logit_pred)
logit_cm    <- confusionMatrix(logit_table)
logit_acc   <- logit_cm$overall[1]

# 
# # Forward
# glm_fit_forward <- stepAIC(glm_fit,
#                             direction = "both",
#                             trace = FALSE)
# glm_probs_forward <- predict(glm_fit_forward,
#                               newdata = test[, -ncol(test)],
#                               type = "response")
# glm_pred_forward <- ifelse(glm_probs_forward > 0.5, 1, 0)
# pred_forward <- sum(glm_pred_forward == test$winner) / length(glm_pred_forward)
# 
# 
# # Backward
# glm_fit_backward <- stepAIC(glm_fit,
#                             direction = "both",
#                             trace = FALSE)
# glm_probs_backward <- predict(glm_fit_backward,
#                               newdata = test[, -ncol(test)],
#                               type = "response")
# glm_pred_backward <- ifelse(glm_probs_backward > 0.5, 1, 0)
# pred_backward <- sum(glm_pred_backward == test$winner) / length(glm_pred_backward)
# 
# 
# # Both
# glm_fit_both <- stepAIC(glm_fit,
#                         direction = "both",
#                         trace = FALSE)
# glm_probs_both <- predict(glm_fit_both,
#                           newdata = test[, -ncol(test)],
#                           type = "response")
# glm_pred_both <- ifelse(glm_probs_both > 0.5, 1, 0)
# pred_both <- sum(glm_pred_both == test$winner) / length(glm_pred_both)
# 
# 
# # ---- Test Models
# summary(glm_fit)
# summary(glm_fit_backward)
# summary(glm_fit_forward)
# summary(glm_fit_both)
# 
# results_df <- data_frame(model    = c("base", "forward", "backward", "both"),
#                          accuracy = c(pred_base, pred_forward, pred_backward, pred_both))
# view(results_df)
# 
# 
# log_base_cm <- table(test$winner, logit_pred)
# acc <- confusionMatrix(log_base_cm)
# acc$overall[1]

# ---- Naive Bayes Models ----
naive_model <- naiveBayes(winner ~ .,
                          data = train_scaled)

naive_pred  <- predict(naive_model,
                       newdata = test_scaled)

naive_table <- table(test_scaled$winner, naive_pred)
naive_cm    <- confusionMatrix(naive_table)
naive_acc   <- naive_cm$overall[1]


# ---- KNN ----
knn_pred <- knn(train = train_scaled[, -ncol(train_scaled)],
                test  = test_scaled[, -ncol(test_scaled)],
                cl    = train_scaled[,  ncol(train_scaled)],
                k     = 11)

knn_table <- table(test_scaled$winner, knn_pred)
knn_cm    <- confusionMatrix(knn_table)
knn_acc   <- knn_cm$overall[1]


# ---- Support Vector Machine ----
svm_model <- svm(formula = winner ~ .,
                 data    = train_scaled)

svm_pred  <- predict(svm_model,
                     newdata = test_scaled[, -ncol(test_scaled)])

svm_table <- table(test_scaled$winner, svm_pred)
svm_cm    <- confusionMatrix(svm_table)
svm_acc   <- svm_cm$overall[1]


# ---- Decision Tree ----
tree_model <- rpart(formula = winner ~ .,
                    data    = train_scaled,
                    cp = 0.07444)

tree_pred <- predict(tree_model,
                     newdata = test_scaled[, -ncol(test_scaled)],
                     type = 'class')

tree_table <- table(test_scaled$winner, tree_pred)
tree_cm    <- confusionMatrix(tree_table)
tree_acc   <- tree_cm$overall[1]

# ---- Model Comparison ----
results_df <-
  data.frame(model    = c("logistic", "naive_bayes", "knn", "svm", "decision_tree"),
             accuracy = c(logit_acc, naive_acc, knn_acc, svm_acc, tree_acc)) %>%
  arrange(desc(accuracy))


# ---- IML ----
features <- as.data.frame(box_score[, !names(box_score) %in% c("game_id", "posteam", "winner_name", "winner")])
response <- as.numeric(as.vector(box_score$winner))
pred <- function(model, newdata) {
  results <- as.data.frame(predict(model, newdata, type = "response"))
  return(results)
}

pred(logit_fit, features)

predictor.glm <- Predictor$new(
  model       = logit_fit,
  data        = features,
  y           = response,
  predict.fun = pred,
  class       = "classification"
)
str(predictor.glm)

# Feature Importance
imp.glm <- FeatureImp$new(predictor.glm, loss = "mse")
plot(imp.glm) + 
  theme_classic() +
  ggtitle("Logistic Regression Feature Importance")

# Feature Interaction
interact.glm <- Interaction$new(predictor.glm) %>% 
  plot()
interact.glm +
  theme_classic() +
  ggtitle("Logistic Regression Feature Interaction")

interact.ry <- Interaction$new(predictor.glm, feature = "receiving_yards_name") %>% 
  plot()
interact.ry +
  theme_classic() +
  ggtitle("Logistic Regression Feature Interaction with Receiving Yards")

# Shapley Values

# ptm <- proc.time()

high  <- predict(logit_fit, features) %>% 
  as.vector() %>% 
  which.max()

highest_prob_win <- features[high, ]

shapley.glm <- Shapley$new(predictor.glm, x.interest = highest_prob_win)
sum(shapley.glm$results["phi"])
plot(shapley.glm) +
  theme_classic()
+
  ggtitle("Shapley Values for TEN vs JAX (42-20)")


proc.time() - ptm









