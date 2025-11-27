# ===================================================================
# 任務模塊 B: LightGBM Stacking 輸入文件生成腳本 
# 目標:
# 1. 解決 feature_pre_filter 參數衝突導致的錯誤。
# 2. 在正式訓練前，加入超參數隨機搜索，為新特徵集尋找最佳參數。
# 3. 採用與 XGBoost 腳本完全一致的特徵工程。
# 4. 使用找到的最佳參數，執行5折交叉驗證，生成 OOF 和 Submission。
# ===================================================================

# --- 1. 環境準備 ---
library(lightgbm)
library(dplyr)
library(readr)
library(caret)
library(recipes)
library(MLmetrics)

print("LightGBM Stacking 輸入文件生成腳本準備就緒！")

# --- 2. 設置、讀取數據與特徵工程 ---
# !!! 注意：請將路徑修改為你自己的實際工作路徑 !!!
setwd("E:/360MoveData/Users/DELL/Desktop/NTU/Analytics software/kaggle/stacking")
print(paste("工作路徑已設置為:", getwd()))

train_df_raw <- readr::read_csv("train.csv")
test_df_raw  <- readr::read_csv("test.csv")
print("數據讀取成功。")

# -- 2.1 特徵工程 (與你的 XGBoost 腳本完全保持一致) --
print("--- 正在創建新特徵 (與XGBoost腳本同步) ---")
train_featured <- train_df_raw %>%
  mutate(
    Edema_numeric = recode(Edema, 'N' = 0, 'S' = 0.5, 'Y' = 1.0),
    Mayo_risk_score = 0.0307*(Age/365.25) + 0.8707*log(Bilirubin) + 0.8592*Edema_numeric - 2.533*log(Albumin) + 0.0388*log(Platelets),
    ALBI_score = (0.66 * log10(Bilirubin)) - (0.085 * Albumin),
    ALBI_status = factor(case_when(ALBI_score <= -2.60 ~ 1, ALBI_score > -2.60 & ALBI_score < 1.39 ~ 2, ALBI_score >= 1.39 ~ 3, TRUE ~ 0))
  ) %>%
  select(-Edema_numeric)

test_featured <- test_df_raw %>%
  mutate(
    Edema_numeric = recode(Edema, 'N' = 0, 'S' = 0.5, 'Y' = 1.0),
    Mayo_risk_score = 0.0307*(Age/365.25) + 0.8707*log(Bilirubin) + 0.8592*Edema_numeric - 2.533*log(Albumin) + 0.0388*log(Platelets),
    ALBI_score = (0.66 * log10(Bilirubin)) - (0.085 * Albumin),
    ALBI_status = factor(case_when(ALBI_score <= -2.60 ~ 1, ALBI_score > -2.60 & ALBI_score < 1.39 ~ 2, ALBI_score >= 1.39 ~ 3, TRUE ~ 0))
  ) %>%
  select(-Edema_numeric)

# -- 2.2 數據預處理 (Recipes) --
recipe_spec <- recipe(Status ~ ., data = train_featured) %>%
  update_role(id, new_role = "ID") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

prepared_recipe <- prep(recipe_spec)
X_processed <- bake(prepared_recipe, new_data = train_featured) %>% select(-id, -Status)
test_processed <- bake(prepared_recipe, new_data = test_featured) %>% select(-id)
y <- as.integer(as.factor(train_df_raw$Status)) - 1
num_class <- length(unique(y))
print("特徵工程與預處理完畢。")

# --- 3. 超參數隨機搜索 ---
print("--- 開始為新特徵集進行超參數隨機搜索 ---")

# -- 3.1 準備數據與搜索空間 --
dtrain_full <- lgb.Dataset(data = as.matrix(X_processed), label = y)
param_grid <- list(
  learning_rate = c(0.005, 0.01, 0.02),
  num_leaves = c(31, 63, 127),
  min_data_in_leaf = c(50, 100, 150),
  feature_fraction = c(0.6, 0.7, 0.8),
  bagging_fraction = c(0.8, 0.9),
  lambda_l1 = c(0.1, 0.5, 1),
  lambda_l2 = c(1, 5, 10)
)
num_searches <- 30 # 執行30次隨機搜索，可以根據時間調整
results_df <- data.frame()

# -- 3.2 執行搜索循環 --
set.seed(123)
for(i in 1:num_searches) {
  cat(sprintf("隨機搜索進度: %d/%d...\n", i, num_searches))
  
  params_sample <- lapply(param_grid, sample, 1)
  
  # 【核心修正】在這裡加入 feature_pre_filter = FALSE
  params_cv <- c(
    list(objective = "multiclass", 
         num_class = num_class, 
         metric = "multi_logloss", 
         bagging_freq = 5, 
         verbose = -1,
         feature_pre_filter = FALSE), # <-- 修正點
    params_sample
  )
  
  cv_model <- lgb.cv(
    params = params_cv,
    data = dtrain_full,
    nrounds = 2000,
    nfold = 5,
    early_stopping_rounds = 50,
    stratified = TRUE
  )
  
  best_score <- cv_model$best_score
  best_iter <- cv_model$best_iter
  
  results_df <- rbind(results_df, data.frame(c(params_sample, list(best_score = best_score, best_iter = best_iter))))
}

# -- 3.3 選出最佳參數 --
best_params_row <- results_df[which.min(results_df$best_score), ]
print("--- 隨機搜索完成 ---")
print("找到的最佳 LogLoss:")
print(best_params_row$best_score)
print("找到的最佳參數組合:")
print(best_params_row)

# --- 4. 使用最佳參數執行5折交叉驗證 ---
params_lgbm_best_list <- as.list(best_params_row[1, !(names(best_params_row) %in% c("best_score", "best_iter"))])

# 在最終訓練時也加入 feature_pre_filter = FALSE
params_lgbm_final <- c(
  list(objective = "multiclass", 
       num_class = num_class, 
       metric = "multi_logloss", 
       bagging_freq = 5, 
       verbose = -1,
       feature_pre_filter = FALSE), # <-- 修正點
  params_lgbm_best_list
)
best_iter_final <- best_params_row$best_iter

set.seed(42)
folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)
oof_preds <- matrix(0, nrow = nrow(X_processed), ncol = num_class)
test_preds_matrix <- array(0, dim = c(nrow(test_processed), num_class, 5))

cat("--- 開始使用調優後的參數進行最終的5折交叉驗證 ---\n")
for (i in 1:5) {
  cat(sprintf("正在處理第 %d/5 折...\n", i))
  
  val_idx <- folds[[i]]
  train_idx <- setdiff(1:nrow(X_processed), val_idx)
  
  dtrain <- lgb.Dataset(data = as.matrix(X_processed[train_idx, ]), label = y[train_idx])
  
  model <- lgb.train(params = params_lgbm_final, data = dtrain, nrounds = best_iter_final)
  
  oof_preds[val_idx, ] <- matrix(predict(model, as.matrix(X_processed[val_idx, ])), ncol = num_class, byrow = FALSE)
  test_preds_matrix[, , i] <- matrix(predict(model, as.matrix(test_processed)), ncol = num_class, byrow = FALSE)
}
cat("交叉驗證和預測生成完畢。\n\n")

# --- 5. 計算並保存結果 ---
oof_logloss <- MultiLogLoss(y_pred = oof_preds, y_true = y)
cat("--- LightGBM (調優後) 模型性能 ---\n")
cat("基於5折交叉驗證的 Out-of-Fold LogLoss:", oof_logloss, "\n\n")

oof_df <- as.data.frame(oof_preds)
colnames(oof_df) <- paste0("lightgbm_oof_", levels(as.factor(train_df_raw$Status)))
oof_df <- oof_df %>% mutate(id = train_df_raw$id, Status = train_df_raw$Status) %>% select(id, Status, everything())
readr::write_csv(oof_df, "oof_train_lgbm_tuned.csv")
print("OOF 文件 'oof_train_lgbm_tuned.csv' 已成功生成。")

test_preds_avg <- as.data.frame(apply(test_preds_matrix, c(1, 2), mean))
colnames(test_preds_avg) <- paste0("Status_", levels(as.factor(train_df_raw$Status)))
submission_df <- test_preds_avg %>% mutate(id = test_df_raw$id) %>% select(id, everything())
readr::write_csv(submission_df, "submission_lgbm_tuned.csv")
print("Submission 文件 'submission_lgbm_tuned.csv' 已成功生成。")
print("===================================================================")

