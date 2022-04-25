# ---- Load Packages ----
# library(conflicted)
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
raw_data <- nflfastR::load_pbp(1999:2021)
raw_data <- clean_pbp(raw_data)
raw_data <- raw_data[complete.cases(raw_data$posteam), ]

schedule <- fast_scraper_schedules(1999:2021) %>% 
  mutate(winner_name = case_when(home_result > 0 ~ home_team,
                                 home_result < 0 ~ away_team,
                                 TRUE            ~ "TIE"))


# ---- Prepare and Partition Data ----
season_stats <- raw_data %>% 
  group_by(game_id, posteam) %>% 
  summarize_at(vars(yards_gained, passing_yards, rushing_yards, epa, interception, fumble, penalty),
               sum, # make sure this doesn't break
               na.rm = TRUE) %>% 
  merge(schedule[, c("game_id", "season", "winner_name", "home_team", "away_team")], by = "game_id", all.x = TRUE)

cumulative_stats <- season_stats %>% 
  group_by(posteam, season) %>% 
  mutate(yards_gained = cumsum(yards_gained),
         yards_gained = lag(yards_gained),
         
         passing_yards = cumsum(passing_yards),
         passing_yards = lag(passing_yards),
         
         rushing_yards = cumsum(rushing_yards),
         rushing_yards = lag(rushing_yards),
         
         epa = cumsum(epa),
         epa = lag(epa),
         
         interception = cumsum(interception),
         interception = lag(interception),
         
         fumble = cumsum(fumble),
         fumble = lag(fumble),
         
         penalty = cumsum(penalty),
         penalty = lag(penalty)) %>% 
  drop_na()

home_team_stats <- cumulative_stats %>% 
  ungroup() %>% 
  filter(posteam == home_team) %>% 
  dplyr::select(-c(season, home_team, away_team))

away_team_stats <- cumulative_stats %>% 
  ungroup() %>% 
  filter(posteam == away_team) %>% 
  dplyr::select(-c(season, home_team, away_team))

model_stats <- home_team_stats %>% 
  merge(away_team_stats, by = c("game_id", "winner_name")) %>% 
  mutate(winner = factor(case_when(posteam.x == winner_name ~ 1,
                                   posteam.y == winner_name ~ 0))) %>% 
  dplyr::select(-c(winner_name, posteam.x, posteam.y)) %>% 
  drop_na()

set.seed(123)
smp_size  <- floor(0.75 * nrow(model_stats))
train_ind <- sample(seq_len(nrow(model_stats)), size = smp_size)

train_not_scaled <- model_stats[train_ind,  -1] %>% 
  drop_na()
test_not_scaled  <- model_stats[-train_ind, -1] %>% 
  drop_na()

train <- train_not_scaled
test  <- test_not_scaled
train[, -ncol(train)] <- scale(train[, -ncol(train)])
test[, -ncol(test)]   <- scale(test[, -ncol(test)])

# ---- Logistic Regression ----
logit_fit   <- glm(winner ~ .,
                 data   = train,
                 family = binomial)
logit_probs <- predict(logit_fit,
                     newdata = test,
                     type    = "response")
logit_pred  <- ifelse(logit_probs > 0.5, 1, 0)

logit_table <- table(test$winner, logit_pred)
logit_cm    <- caret::confusionMatrix(logit_table)
logit_acc   <- logit_cm$overall[1]


# ---- Naive Bayes ----
naive_model <- naiveBayes(winner ~ .,
                          data = train)

naive_pred  <- predict(naive_model,
                       newdata = test)

naive_table <- table(test$winner, naive_pred)
naive_cm    <- caret::confusionMatrix(naive_table)
naive_acc   <- naive_cm$overall[1]


# ---- KNN ----
knn_pred <- knn(train = train,
                test  = test,
                cl    = train$winner,
                k     = 11)

knn_table <- table(test$winner, knn_pred)
knn_cm    <- caret::confusionMatrix(knn_table)
knn_acc   <- knn_cm$overall[1]


# ---- Support Vector Machine ----
svm_model <- svm(formula = winner ~ .,
                 data    = train)

svm_pred  <- predict(svm_model,
                     newdata = test)

svm_table <- table(test$winner, svm_pred)
svm_cm    <- caret::confusionMatrix(svm_table)
svm_acc   <- svm_cm$overall[1]


# ---- Decision Tree ----
tree_model <- rpart(formula = winner ~ .,
                    data    = train,
                    cp = 0.07444)

tree_pred <- predict(tree_model,
                     newdata = test,
                     type = 'class')

tree_table <- table(test$winner, tree_pred)
tree_cm    <- caret::confusionMatrix(tree_table)
tree_acc   <- tree_cm$overall[1]


# ---- Vegas ----
vegas_results_df <- model_stats[-train_ind, ] %>% 
  merge(raw_data[, c("game_id", "spread_line")], by = "game_id", all.x = TRUE) %>% 
  drop_na() %>% 
  distinct() %>% 
  mutate(spread = factor(case_when(spread_line < 0 ~ 0,
                                   spread_line > 0 ~ 1)))


vegas_table <- table(test$winner, vegas_results_df$spread)
vegas_cm    <- caret::confusionMatrix(vegas_table)
vegas_acc   <- vegas_cm$overall[1]


# ---- Model Comparison ----
results_df <-
  data.frame(model    = c("logistic", "naive_bayes", "knn", "svm", "decision_tree", "vegas"),
             accuracy = c(logit_acc, naive_acc, knn_acc, svm_acc, tree_acc, vegas_acc)) %>%
  arrange(desc(accuracy))
view(results_df)

# ---- IML ----
features <- as.data.frame(model_stats[, !names(model_stats) %in% c("game_id", "winner")])
response <- as.numeric(as.vector(model_stats$winner))
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

interact.ry <- Interaction$new(predictor.glm, feature = "yards_gained.x") %>% 
  plot()
interact.ry +
  theme_classic() +
  ggtitle("Logistic Regression Feature Interaction with Receiving Yards")

# Shapley Values
high <- predict(logit_fit, features) %>% 
  as.vector() %>% 
  which.max()

highest_prob_win <- features[high, ]

shapley.glm <- Shapley$new(predictor.glm, x.interest = highest_prob_win)
sum(shapley.glm$results["phi"])
plot(shapley.glm) +
  theme_classic()

