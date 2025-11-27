# ===================================================================
# 最終模塊: Stacking 模型整合與預測 (V3)
# 核心升級:
# 1. 【模型還原】恢復使用更強大的 XGBoost 作為元模型。
# 2. 【元特徵】創建關於基模型預測的統計特徵 (平均值、標準差等)。
# 3. 【最終融合】將 Stacking 的結果與最佳單模型進行加權平均。
# ===================================================================

# --- 1. 環境準備 ---
library(dplyr)
library(readr)
library(xgboost)
library(MLmetrics)
library(matrixStats) # 用於計算行標準差

print("最終 Stacking 整合腳本 (V3 - 終極版) 準備就緒！")

# --- 2. 設置與加載數據 ---
# !!! 注意：請將路徑修改為你自己的實際工作路徑 !!!
setwd("E:/360MoveData/Users/DELL/Desktop/NTU/Analytics software/kaggle/stacking")
print(paste("工作路徑已設置為:", getwd()))

# -- 2.1 讀取原始數據 --
train_df_raw <- readr::read_csv("train.csv")
test_df_raw  <- readr::read_csv("test.csv")
print("原始數據 train.csv 和 test.csv 加載成功。")

# -- 2.2 定義文件名 --
oof_files <- c(
  xgb = "oof_train_xgb.csv",
  lgbm = "oof_train_lgbm.csv",
  cat = "oof_train_cat.csv"
)

submission_files <- c(
  xgb = "submission_xgb.csv",
  lgbm = "submission_lgbm.csv",
  cat = "submission_cat.csv"
)

# -- 2.3 檢查文件是否存在 --
all_files <- c(oof_files, submission_files)
if(!all(file.exists(all_files))){
  stop("錯誤：缺少一個或多個基模型的 .csv 結果文件！")
}

# -- 2.4 加載所有數據 --
oof_xgb <- readr::read_csv(oof_files["xgb"]); oof_lgbm <- readr::read_csv(oof_files["lgbm"]); oof_cat <- readr::read_csv(oof_files["cat"])
sub_xgb <- readr::read_csv(submission_files["xgb"]); sub_lgbm <- readr::read_csv(submission_files["lgbm"]); sub_cat <- readr::read_csv(submission_files["cat"])
print("所有基模型的 OOF 和 Submission 文件加載成功。")


# --- 3. 創建元模型的訓練集和測試集 ---
print("--- 正在創建元模型的訓練數據 ---")

# -- 3.1 準備 Level-1 特徵 (基模型預測) --
oof_probs_xgb <- oof_xgb %>% select(starts_with("xgb_oof_"))
oof_probs_lgbm <- oof_lgbm %>% select(starts_with("lightgbm_oof_"))
oof_probs_cat <- oof_cat %>% select(starts_with("oof_catboost_"))
meta_X_train_lvl1 <- bind_cols(oof_probs_xgb, oof_probs_lgbm, oof_probs_cat)

test_probs_xgb <- sub_xgb %>% select(starts_with("Status_"))
test_probs_lgbm <- sub_lgbm %>% select(starts_with("Status_"))
test_probs_cat <- sub_cat %>% select(starts_with("Status_"))
meta_X_test_lvl1 <- bind_cols(test_probs_xgb, test_probs_lgbm, test_probs_cat)
colnames(meta_X_test_lvl1) <- colnames(meta_X_train_lvl1)

# -- 3.2 【新增功能】創建元特徵 (Meta-Features) --
print("--- 正在創建元特徵 (預測的平均值、標準差等) ---")
# 訓練集元特徵
meta_X_train_meta <- data.frame(
  meta_C_mean = rowMeans(select(meta_X_train_lvl1, ends_with("_C"))),
  meta_C_sd = rowSds(as.matrix(select(meta_X_train_lvl1, ends_with("_C")))),
  meta_CL_mean = rowMeans(select(meta_X_train_lvl1, ends_with("_CL"))),
  meta_CL_sd = rowSds(as.matrix(select(meta_X_train_lvl1, ends_with("_CL")))),
  meta_D_mean = rowMeans(select(meta_X_train_lvl1, ends_with("_D"))),
  meta_D_sd = rowSds(as.matrix(select(meta_X_train_lvl1, ends_with("_D"))))
)
# 測試集元特徵
meta_X_test_meta <- data.frame(
  meta_C_mean = rowMeans(select(meta_X_test_lvl1, ends_with("_C"))),
  meta_C_sd = rowSds(as.matrix(select(meta_X_test_lvl1, ends_with("_C")))),
  meta_CL_mean = rowMeans(select(meta_X_test_lvl1, ends_with("_CL"))),
  meta_CL_sd = rowSds(as.matrix(select(meta_X_test_lvl1, ends_with("_CL")))),
  meta_D_mean = rowMeans(select(meta_X_test_lvl1, ends_with("_D"))),
  meta_D_sd = rowSds(as.matrix(select(meta_X_test_lvl1, ends_with("_D"))))
)

# -- 3.3 準備 Level-0 特徵 (原始特徵) --
important_orig_features <- c("Age", "Bilirubin", "Albumin", "Copper", "Prothrombin")
meta_X_train_lvl0 <- train_df_raw[, important_orig_features]
meta_X_test_lvl0 <- test_df_raw[, important_orig_features]

# -- 3.4 合併所有特徵 --
meta_X_train <- bind_cols(meta_X_train_lvl1, meta_X_train_meta, meta_X_train_lvl0)
meta_X_test <- bind_cols(meta_X_test_lvl1, meta_X_test_meta, meta_X_test_lvl0)

# -- 3.5 準備目標 (y_train) --
meta_y_train <- as.factor(train_df_raw$Status)
meta_y_train_numeric <- as.integer(meta_y_train) - 1
num_class <- length(levels(meta_y_train))
print("元模型數據準備完畢。")
print(paste("元模型訓練集維度:", dim(meta_X_train)[1], "行,", dim(meta_X_train)[2], "個特徵"))


# --- 4. 使用 CV 調優並訓練 XGBoost 元模型 ---
print("--- 正在使用交叉驗證為元模型尋找最佳迭代次數... ---")
dtrain_meta <- xgb.DMatrix(data = as.matrix(meta_X_train), label = meta_y_train_numeric)
params_meta <- list(objective="multi:softprob", eval_metric="mlogloss", num_class=num_class, eta=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8)

set.seed(123)
cv_results <- xgb.cv(params=params_meta, data=dtrain_meta, nrounds=500, nfold=5, early_stopping_rounds=20, verbose=0)
best_iteration <- cv_results$best_iteration
print(paste("CV找到的最佳迭代次數為:", best_iteration))

print("--- 正在訓練最終的 XGBoost 元模型... ---")
set.seed(123)
meta_model_xgb <- xgb.train(params=params_meta, data=dtrain_meta, nrounds=best_iteration, verbose=0)
print("XGBoost 元模型訓練完成！")


# --- 5. 生成 Stacking 預測 ---
print("--- 正在生成 Stacking 模型的預測... ---")
dtest_meta <- xgb.DMatrix(data = as.matrix(meta_X_test))
stacking_preds_prob <- predict(meta_model_xgb, dtest_meta, reshape = TRUE)
stacking_submission_df <- as.data.frame(stacking_preds_prob) %>%
  mutate(id = test_df_raw$id) %>%
  select(id, everything())
colnames(stacking_submission_df) <- c("id", "Status_C", "Status_CL", "Status_D")
# readr::write_csv(stacking_submission_df, "submission_stacking_only.csv") # 可選：保存純stacking結果


# --- 6. 最終融合：將 Stacking 結果與最佳單模型加權平均 ---
print("--- 正在執行最終融合 (Stacking + 最佳單模型)... ---")

# !!! 核心調參區 !!!
# blending_weight: Stacking 結果的權重，(1-blending_weight) 是最佳單模型的權重
blending_weight <- 0.55
best_single_model_submission <- sub_xgb # <--  XGBoost 是最好的單模型

# 確保 id 順序一致
best_single_model_submission <- best_single_model_submission %>% arrange(id)
stacking_submission_df <- stacking_submission_df %>% arrange(id)

# 提取概率矩陣
prob_stack <- as.matrix(stacking_submission_df[, -1])
prob_single <- as.matrix(best_single_model_submission[, -1])

# 執行加權平均
final_blended_probs <- blending_weight * prob_stack + (1 - blending_weight) * prob_single

# --- 7. 生成最終提交文件 ---
final_submission_df <- as.data.frame(final_blended_probs)
colnames(final_submission_df) <- c("Status_C", "Status_CL", "Status_D")
final_submission_df <- final_submission_df %>%
  mutate(id = test_df_raw$id) %>%
  select(id, everything())

readr::write_csv(final_submission_df, "submission_stacking_ultimate.csv")

print("===================================================================")
print("終極版提交文件 'submission_stacking_ultimate.csv' 已成功生成！")
print(paste("本次融合中，Stacking模型的權重為", blending_weight))
print("===================================================================")

