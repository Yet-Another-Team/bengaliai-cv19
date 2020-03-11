args <- commandArgs(trailingOnly = TRUE)
experiment1OutPath <- args[1]
experiment2OutPath <- args[2]
outFile <- args[3]
maxMetricToLoadColumn <- "val_root_recall" # which metric was maximized during training, thus last checkpoint corresponds to max of this metirc
metricToEvaluateColumn <- args[4]

exp1_runs <- as.numeric(dir(experiment1OutPath))
exp1_runs <- exp1_runs[!is.na(exp1_runs)]
exp2_runs <- as.numeric(dir(experiment2OutPath))
exp2_runs <- exp2_runs[!is.na(exp2_runs)]
exp_runs <- intersect(exp1_runs,exp2_runs)

a = c()
b = c()
for(exp_run in exp_runs) {
  df1 <- read.csv(file.path(experiment1OutPath,exp_run,'training_log.csv'))  
  metric1 <- df1[[metricToEvaluateColumn]][which.max(df1[[maxMetricToLoadColumn]])]
  a = c(a,metric1)
  df2 <- read.csv(file.path(experiment2OutPath,exp_run,'training_log.csv'))
  metric2 <-  df2[[metricToEvaluateColumn]][which.max(df2[[maxMetricToLoadColumn]])]
  b = c(b,metric2)
}

Model1 <- a
Model2 <- b



t_res <- t.test(Model1,Model2,paired = T)
sink(outFile);
print(paste0("Metric that was maximized during training ",maxMetricToLoadColumn))
print(paste0("Metric T-tested ",metricToEvaluateColumn))
print(Model1)
print("Model 2 metric values:")
print(Model2)
print(t_res);
sink() 
print(paste0("Metric that was maximized during training ",maxMetricToLoadColumn))
print(paste0("Metric T-tested ",metricToEvaluateColumn))
print("Model 1 metric values:")
print(Model1)
print("Model 2 metric values:")
print(Model2)
print(t_res)
