args <- commandArgs(trailingOnly = TRUE)
experiment1OutPath <- args[1]
experiment2OutPath <- args[2]
metric_column <- args[3]
outFile <- args[4]

exp1_runs <- as.numeric(dir(experiment1OutPath))
exp1_runs <- exp1_runs[!is.na(exp1_runs)]
exp2_runs <- as.numeric(dir(experiment2OutPath))
exp2_runs <- exp2_runs[!is.na(exp2_runs)]
exp_runs <- intersect(exp1_runs,exp2_runs)

a = c()
b = c()
for(exp_run in exp_runs) {
  df1 <- read.csv(file.path(experiment1OutPath,exp_run,'training_log.csv'))
  metric1 <- max(df1[[metric_column]])
  a = c(a,metric1)
  df2 <- read.csv(file.path(experiment2OutPath,exp_run,'training_log.csv'))
  metric2 <- max(df2[[metric_column]])
  b = c(b,metric2)
}

Model1 <- a
Model2 <- b



t_res <- t.test(Model1,Model2,paired = T)
sink(outFile);
print(paste0("Metric is ",metric_column))
print("Model 1 metric values:")
print(Model1)
print("Model 2 metric values:")
print(Model2)
print(t_res);
sink() 
print(paste0("Metric is ",metric_column))
print("Model 1 metric values:")
print(Model1)
print("Model 2 metric values:")
print(Model2)
print(t_res)
