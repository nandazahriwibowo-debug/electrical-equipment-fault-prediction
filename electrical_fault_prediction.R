# ============================================================================
# Electrical equipment fault prediction
# Author: Nanda Zahri Wibowo
# CYO Project - edX Data Science
# ============================================================================
# This is a complete, self-contained R script for Electrical Equipment 
# Fault Prediction using the AI4I 2020 Predictive Maintenance Dataset.
# The script runs without errors and is fully reproducible.
# ============================================================================

# 1. SETUP & PACKAGE INSTALLATION
# ============================================================================

# Auto-install missing packages - robust for CYO / edX submission
required_packages <- c(
  "tidyverse", "caret", "corrplot", "pROC", "knitr", "kableExtra",
  "ggplot2", "dplyr", "readr", "gbm", "randomForest", "scales", "gridExtra"
)

install_if_missing <- function(p) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", quiet = TRUE)
    library(p, character.only = TRUE)
  }
}
invisible(sapply(required_packages, install_if_missing))

# Load libraries
library(tidyverse)
library(caret)
library(corrplot)
library(pROC)
library(ggplot2)
library(randomForest)
library(gbm)

# Set global seed for full reproducibility
set.seed(2025)

# 2. DATA LOADING
# ============================================================================
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# Source: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
# License: CC BY 4.0

data_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
data_path <- "ai4i2020.csv"

if (!file.exists(data_path)) {
  download.file(data_url, destfile = data_path, mode = "wb", quiet = TRUE)
  cat("Dataset downloaded successfully.\n")
}

# Load data with readr for robust parsing
raw_data <- readr::read_csv(data_path, show_col_types = FALSE)

# Standardize column names for R compatibility
colnames(raw_data) <- c(
  "UDI", "Product_ID", "Type", 
  "Air_temperature_K", "Process_temperature_K",
  "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
  "Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"
)

# Convert categorical variables to factors
raw_data$Type <- as.factor(raw_data$Type)
raw_data$Machine_failure <- factor(raw_data$Machine_failure, levels = c(0,1), labels = c("No_Failure", "Failure"))

cat("Dataset Dimensions: ", dim(raw_data), "\n")
print(dplyr::glimpse(raw_data))
print(summary(raw_data))

# 3. EXPLORATORY DATA ANALYSIS & DATA CLEANING
# ============================================================================

# 3.1 Missing Values Audit
missing_summary <- colSums(is.na(raw_data))
missing_df <- data.frame(
  Variable = names(missing_summary),
  Missing_Count = missing_summary,
  Missing_Percent = round(missing_summary / nrow(raw_data) * 100, 2)
)
print(missing_df)

# Defensive median imputation stub (dataset has 0 NAs)
clean_data <- raw_data
numeric_cols <- sapply(clean_data, is.numeric)
for (col in names(clean_data)[numeric_cols]) {
  if (any(is.na(clean_data[[col]]))) {
    clean_data[[col]][is.na(clean_data[[col]])] <- median(clean_data[[col]], na.rm = TRUE)
  }
}

# 3.2 Duplicate Check
dup_count <- sum(duplicated(clean_data))
cat("Number of duplicate rows detected:", dup_count, "\n")
if (dup_count > 0) {
  clean_data <- clean_data[!duplicated(clean_data), ]
}

# 3.3 Outlier Treatment - Winsorization
# Winsorization function - caps extreme 1% tails
winsorize <- function(x, probs = c(0.01, 0.99)) {
  bounds <- quantile(x, probs = probs, na.rm = TRUE)
  x[x < bounds[1]] <- bounds[1]
  x[x > bounds[2]] <- bounds[2]
  return(x)
}

sensor_vars <- c("Air_temperature_K", "Process_temperature_K", 
                 "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min")

clean_data_w <- clean_data
clean_data_w[sensor_vars] <- lapply(clean_data_w[sensor_vars], winsorize)
clean_data <- clean_data_w
cat("Outlier capping complete. Winsorization applied at 1%/99%.\n")

# 3.4 EDA Visualizations

# Visualization 1: Class Imbalance – Failure Distribution
viz1 <- ggplot(clean_data, aes(x = Machine_failure, fill = Machine_failure)) +
  geom_bar() +
  scale_fill_manual(values = c("No_Failure" = "#59A14F", "Failure" = "#E15759")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  theme_minimal(base_size = 13) +
  labs(title = "Electrical Equipment Failure Class Distribution",
       x = "Machine Status", y = "Count") +
  theme(legend.position = "none")
print(viz1)
ggsave("viz1_failure_dist.png", viz1, width = 7, height = 5, dpi = 150)

# Visualization 2: Correlation Heatmap
numeric_data <- clean_data %>% 
  dplyr::select(Air_temperature_K, Process_temperature_K,
                Rotational_speed_rpm, Torque_Nm, Tool_wear_min) %>%
  mutate(Machine_failure_num = ifelse(clean_data$Machine_failure == "Failure", 1, 0))

cor_mat <- cor(numeric_data)
png("viz2_correlation.png", width = 800, height = 700)
corrplot::corrplot(cor_mat, method = "color", type = "upper", 
                   tl.col = "black", tl.srt = 45,
                   addCoef.col = "black", number.cex = 0.7,
                   title = "Sensor Correlation Matrix", mar=c(0,0,2,0))
dev.off()

# Visualization 3: Torque Distribution by Failure Status
viz3 <- ggplot(clean_data, aes(x = Machine_failure, y = Torque_Nm, fill = Machine_failure)) +
  geom_boxplot(alpha = 0.8, outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.05, size = 0.5) +
  scale_fill_manual(values = c("No_Failure" = "#76B7B2", "Failure" = "#E15759")) +
  theme_minimal(base_size = 13) +
  labs(title = "Torque Load Distribution by Equipment Failure Status",
       x = "Equipment Status", y = "Torque [Nm]") +
  theme(legend.position = "none")
print(viz3)
ggsave("viz3_torque.png", viz3, width = 7, height = 5, dpi = 150)

# Visualization 4: Rotational Speed vs. Torque – Failure Map
viz4 <- ggplot(clean_data, aes(x = Rotational_speed_rpm, y = Torque_Nm, color = Machine_failure)) +
  geom_point(alpha = 0.5, size = 1.8) +
  scale_color_manual(values = c("No_Failure" = "#BAB0AC", "Failure" = "#E15759")) +
  theme_bw(base_size = 13) +
  labs(title = "Operational Envelope: Speed vs Torque",
       subtitle = "Failures cluster in high-torque, low-speed region",
       x = "Rotational Speed [rpm]", y = "Torque [Nm]", color = "Status") +
  theme(legend.position = "bottom")
print(viz4)
ggsave("viz4_speed_torque.png", viz4, width = 7, height = 5, dpi = 150)

# 4. FEATURE ENGINEERING
# ============================================================================
model_data <- clean_data %>%
  mutate(
    # Thermal differential - indicates cooling efficiency
    Temperature_diff_K = Process_temperature_K - Air_temperature_K,
    
    # Mechanical power [W] = Torque [Nm] * Angular velocity [rad/s]
    Power_W = Torque_Nm * Rotational_speed_rpm * 2 * pi / 60,
    
    # Torque-Speed Stress Index
    Torque_speed_ratio = Torque_Nm / Rotational_speed_rpm,
    
    # Tool wear bins - non-linear degradation
    Tool_wear_cat = cut(Tool_wear_min,
                        breaks = c(-Inf, 50, 120, 190, Inf),
                        labels = c("New", "Low", "Medium", "High")),
    
    # High torque flag
    High_torque_flag = factor(ifelse(Torque_Nm > 55, "High", "Normal"))
  ) %>%
  # Drop identifiers and leakage columns
  dplyr::select(-UDI, -Product_ID, -TWF, -HDF, -PWF, -OSF, -RNF)

model_data$Type <- as.factor(model_data$Type)
model_data$Tool_wear_cat <- as.factor(model_data$Tool_wear_cat)

cat("Feature engineering complete. Final features:", ncol(model_data), "\n")

# 5. DATA PARTITIONING
# ============================================================================
# Stratified 80/20 split to preserve failure rate
train_index <- caret::createDataPartition(model_data$Machine_failure, p = 0.80, list = FALSE)
train_set <- model_data[train_index, ]
test_set  <- model_data[-train_index, ]

cat("Training set dimensions:", dim(train_set), "\n")
cat("Test set dimensions:", dim(test_set), "\n")
cat("Train failure rate:", round(mean(train_set$Machine_failure == "Failure")*100, 2), "%\n")
cat("Test failure rate:", round(mean(test_set$Machine_failure == "Failure")*100, 2), "%\n")

# 6. MODELING
# ============================================================================
# Training control with 5-fold CV, optimizing ROC
ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final",
  verboseIter = FALSE
)

# Model 1: Logistic Regression
cat("\nTraining Logistic Regression...\n")
set.seed(2025)
model_glm <- caret::train(
  Machine_failure ~ .,
  data = train_set,
  method = "glm",
  family = "binomial",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  metric = "ROC"
)
print(model_glm)

# Model 2: Random Forest
cat("\nTraining Random Forest...\n")
set.seed(2025)
model_rf <- caret::train(
  Machine_failure ~ .,
  data = train_set,
  method = "rf",
  ntree = 350,
  tuneGrid = data.frame(mtry = c(2,3,4,5)),
  preProcess = c("center", "scale"),
  trControl = ctrl,
  metric = "ROC",
  importance = TRUE
)
print(model_rf)

# Model 3: Gradient Boosting Machine
cat("\nTraining Gradient Boosting Machine...\n")
set.seed(2025)
model_gbm <- caret::train(
  Machine_failure ~ .,
  data = train_set,
  method = "gbm",
  trControl = ctrl,
  metric = "ROC",
  verbose = FALSE,
  tuneGrid = expand.grid(
    n.trees = c(100, 150),
    interaction.depth = c(2, 3),
    shrinkage = 0.1,
    n.minobsinnode = 10
  )
)
print(model_gbm)

# 7. MODEL EVALUATION
# ============================================================================
evaluate_model <- function(model, test_data, model_name) {
  preds_class <- predict(model, newdata = test_data)
  preds_prob <- predict(model, newdata = test_data, type = "prob")[, "Failure"]
  
  cm <- caret::confusionMatrix(preds_class, test_data$Machine_failure, positive = "Failure")
  
  roc_obj <- pROC::roc(response = test_data$Machine_failure,
                       predictor = preds_prob,
                       levels = c("No_Failure", "Failure"),
                       quiet = TRUE)
  
  data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall = as.numeric(cm$byClass["Recall"]),
    F1 = as.numeric(cm$byClass["F1"]),
    AUC_ROC = as.numeric(roc_obj$auc)
  )
}

results_glm <- evaluate_model(model_glm, test_set, "Logistic Regression")
results_rf  <- evaluate_model(model_rf, test_set, "Random Forest")
results_gbm <- evaluate_model(model_gbm, test_set, "Gradient Boosting")

results_all <- rbind(results_glm, results_rf, results_gbm)
print(results_all)

# Save results table
write.csv(results_all, "model_performance_results.csv", row.names = FALSE)

# ROC Curve Comparison
roc_glm <- pROC::roc(test_set$Machine_failure, predict(model_glm, test_set, type="prob")[, "Failure"], quiet=TRUE)
roc_rf  <- pROC::roc(test_set$Machine_failure, predict(model_rf, test_set, type="prob")[, "Failure"], quiet=TRUE)
roc_gbm <- pROC::roc(test_set$Machine_failure, predict(model_gbm, test_set, type="prob")[, "Failure"], quiet=TRUE)

png("roc_comparison.png", width = 800, height = 600)
plot(roc_gbm, col = "#E15759", lwd = 2.5, main = "ROC Curves - Electrical Fault Prediction Models")
plot(roc_rf, col = "#4E79A7", lwd = 2, add = TRUE)
plot(roc_glm, col = "#59A14F", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(
         paste0("GBM AUC = ", round(roc_gbm$auc,3)),
         paste0("RF  AUC = ", round(roc_rf$auc,3)),
         paste0("GLM AUC = ", round(roc_glm$auc,3))
       ),
       col = c("#E15759", "#4E79A7", "#59A14F"), lwd = 2, bty = "n")
dev.off()

# Feature Importance - GBM
png("feature_importance_gbm.png", width = 800, height = 600)
plot(caret::varImp(model_gbm), top = 10, main = "GBM - Top 10 Feature Importance for Fault Prediction")
dev.off()

cat("\n========================================\n")
cat("FINAL MODEL PERFORMANCE SUMMARY\n")
cat("========================================\n")
print(results_all)
cat("\nBest Model: Gradient Boosting Machine\n")
cat("AUC-ROC: ", round(results_gbm$AUC_ROC, 4), "\n")
cat("Recall: ", round(results_gbm$Recall, 4), "\n")
cat("Precision: ", round(results_gbm$Precision, 4), "\n")
cat("\nAll visualizations saved as PNG files.\n")
cat("Project complete - Author: Nanda Zahri Wibowo\n")
