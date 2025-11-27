# ===================================================================
# 任務模塊 A: XGBoost 專業調優與特徵生成腳本
# 工作流:
# 1. 創建新特徵。
# 2. 使用 tidymodels 進行5折交叉驗證，自動尋找最佳超參數。
# 3. 使用找到的最佳參數，在手動5折交叉驗證循環中生成 OOF 特徵。
# ===================================================================

# --- 1. 環境準備 ---
library(tidymodels)
library(xgboost)
library(caret)
library(MLmetrics)
library(dplyr)
library(readr)

print("XGBoost 專業調優與特徵生成腳本環境準備就緒！")

# --- 2. 設置、讀取數據與特徵工程 ---
setwd("E:/360MoveData/Users/DELL/Desktop/NTU/Analytics software/kaggle")
print(paste("工作路徑已設置為:", getwd()))

train_df_raw <- readr::read_csv("train.csv")
test_df_raw  <- readr::read_csv("test.csv")
print("數據讀取成功。")

# -- 2.1 特徵工程 (與隊友保持一致) --
print("--- 正在創建新特徵：ALBI Score 和 Mayo Risk Score ---")
train_featured <- train_df_raw %>%
  mutate(
    Edema_numeric = recode(Edema, 'N' = 0, 'S' = 0.5, 'Y' = 1.0),
    ALBI_score = (log10(Bilirubin * 17.1) * 0.66) + (Albumin * 10 * -0.085),
    Mayo_risk_score = 0.0307*(Age/365.25) + 0.8707*log(Bilirubin) + 0.8592*Edema_numeric - 2.532*log(Albumin) + 0.0031*log(Platelets) + 2.38*log(Prothrombin)
  ) %>%
  select(-Edema_numeric)

test_featured <- test_df_raw %>%
  mutate(
    Edema_numeric = recode(Edema, 'N' = 0, 'S' = 0.5, 'Y' = 1.0),
    ALBI_score = (log10(Bilirubin * 17.1) * 0.66) + (Albumin * 10 * -0.085),
    Mayo_risk_score = 0.0307*(Age/365.25) + 0.8707*log(Bilirubin) + 0.8592*Edema_numeric - 2.532*log(Albumin) + 0.0031*log(Platelets) + 2.38*log(Prothrombin)
  ) %>%
  select(-Edema_numeric)
print("新特徵創建成功！")

# --- 3. 超參數調優：尋找最佳 XGBoost 參數 ---
print("--- 第三步：開始超參數調優以尋找最佳參數 ---")

# -- 3.1 準備 tidymodels 所需的數據和流程 --
train_tm <- train_featured %>% mutate(Status = factor(Status))
set.seed(123)
cv_folds <- vfold_cv(train_tm, v = 5, strata = Status)

recipe_tm <- recipe(Status ~ ., data = train_tm) %>%
  update_role(id, new_role = "ID") %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

spec_tm <- boost_tree(
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune() # mtry 對應 xgboost 的 colsample_bytree
) %>% set_engine("xgboost") %>% set_mode("classification")

wf_tm <- workflow() %>% add_recipe(recipe_tm) %>% add_model(spec_tm)

# -- 3.2 進行網格搜索 --
print("正在進行網格搜索... (這會花費一些時間)")
set.seed(123)
tune_res <- tune_grid(
  wf_tm,
  resamples = cv_folds,
  grid = 20, # 嘗試20組隨機參數組合
  metrics = metric_set(mn_log_loss)
)

# -- 3.3 提取最佳參數 --
best_params <- select_best(tune_res, metric = "mn_log_loss")
print("--- 找到的最佳超參數組合 ---")
print(best_params)

# --- 4. 使用最佳參數，手動生成 OOF 特徵 ---
print("--- 第四步：使用最佳參數生成 OOF 特徵 ---")

y <- as.integer(as.factor(train_featured$Status)) - 1
status_levels <- levels(as.factor(train_featured$Status))
num_class <- length(status_levels)

set.seed(123)
folds <- createFolds(train_featured$Status, k = 5, list = TRUE, returnTrain = FALSE)

oof_train_preds <- matrix(0, nrow = nrow(train_featured), ncol = num_class)
test_preds_matrix <- array(0, dim = c(nrow(test_featured), num_class, 5))

# -- 4.1 將最佳參數智能轉換為 XGBoost 格式 --

# 需要先知道預處理後總共有多少特徵
# 以便將 mtry (特徵數) 轉換為 colsample_bytree (特徵比例)
temp_recipe <- recipe(~ ., data = train_featured %>% select(-id, -Status)) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

temp_prepped <- prep(temp_recipe)
num_total_features <- ncol(bake(temp_prepped, new_data = train_featured))
print(paste("預處理後的總特徵數量為:", num_total_features))

# [修正] 將 mtry (整數) 轉換為 colsample_bytree (0到1之間的比例)
colsample_prop <- best_params$mtry / num_total_features
# 確保比例不會超過1 (安全措施)
if (colsample_prop > 1) { colsample_prop <- 1.0 }
print(paste("最佳 mtry:", best_params$mtry, "轉換為 colsample_bytree 比例:", round(colsample_prop, 3)))

params_xgb_tuned <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class,
  eta = best_params$learn_rate,
  max_depth = best_params$tree_depth,
  min_child_weight = best_params$min_n, # min_n 對應 min_child_weight
  colsample_bytree = colsample_prop,     # <-- 使用修正後的比例！
  subsample = 0.8, # 可以保留一個不錯的固定值
  seed = 123
)

# -- 4.2 執行手動交叉驗證循環 --
for (i in 1:5) {
  cat(paste("--- 正在處理第", i, "/ 5 折 ---\n"))
  
  val_idx <- folds[[i]]
  train_idx <- setdiff(1:nrow(train_featured), val_idx)
  
  # 數據預處理 (獨熱編碼)
  recipe_spec_manual <- recipe(~ ., data = train_featured[train_idx,] %>% select(-id, -Status)) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors())
  
  recipe_prepped <- prep(recipe_spec_manual)
  train_processed <- bake(recipe_prepped, new_data = train_featured[train_idx,])
  val_processed <- bake(recipe_prepped, new_data = train_featured[val_idx,])
  test_processed <- bake(recipe_prepped, new_data = test_featured)
  
  dtrain <- xgb.DMatrix(data = as.matrix(train_processed), label = y[train_idx])
  dval <- xgb.DMatrix(data = as.matrix(val_processed), label = y[val_idx])
  dtest <- xgb.DMatrix(data = as.matrix(test_processed))
  
  # 訓練模型
  model <- xgb.train(
    params = params_xgb_tuned, # <-- 使用調優後的參數！
    data = dtrain,
    nrounds = 2000,
    watchlist = list(eval = dval, train = dtrain),
    early_stopping_rounds = 50,
    print_every_n = 100,
    verbose = 0 # 設為0以簡化日誌
  )
  
  best_iteration <- model$best_iteration
  oof_train_preds[val_idx, ] <- matrix(predict(model, dval, iterationrange = c(1, best_iteration)), ncol = num_class, byrow = TRUE)
  test_preds_matrix[, , i] <- matrix(predict(model, dtest, iterationrange = c(1, best_iteration)), ncol = num_class, byrow = TRUE)
}
cat("交叉驗證和預測生成完毕。\n\n")

# --- 5. 計算並保存結果 ---
oof_logloss <- MultiLogLoss(y_pred = oof_train_preds, y_true = y)
cat("--- 調優後 XGBoost 模型性能 ---\n")
cat("基於5折交叉驗證的 Out-of-Fold LogLoss:", oof_logloss, "\n\n")

oof_probs <- as.data.frame(oof_train_preds)
colnames(oof_probs) <- paste0("xgb_oof_", status_levels)
oof_train_final <- cbind(data.frame(id = train_df_raw$id, Status = train_df_raw$Status), oof_probs)
readr::write_csv(oof_train_final, "oof_train_xgb_tuned.csv")
cat("'oof_train_xgb_tuned.csv' 已生成。\n")

test_preds_avg <- as.data.frame(apply(test_preds_matrix, c(1, 2), mean))
colnames(test_preds_avg) <- paste0("Status_", status_levels)
submission_df <- cbind(data.frame(id = test_df_raw$id), test_preds_avg)
readr::write_csv(submission_df, "submission_xgb_tuned.csv")
cat("'submission_xgb_tuned.csv' 已生成。\n")

print("所有流程完成！")

