# ================================= A test case =================================
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)
library(mice)
library(micd)
library(miceadds)
library(dplyr)
source("R/MeeksRules.R")
source("R/tcheckTriple.R")
source("R/tpc_cons_intern.R")
source("R/tskeleton.R")
source("R/tskeleton_parallel.R")
source("R/tpc.R")
source("R/fixedMItest.R")
source("cuda/cuPCMI.R")

shdsum <- 0
hdsum <- 0
alpha <- 0.01
max_order <- 12

# read data as imputed_data
dataset_path <- file.path("dataset_imputed_60/dataset_imputed_60.Rdata", fsep = .Platform$file.sep)
load.Rdata(dataset_path, "imputed_data")

# make suffStat
suffStatMI <- micd::getSuff(imputed_data, test="gaussMItest")
suffStatSI <- list(C = suffStatMI[[1]], n = suffStatMI[[length(suffStatMI)]])
# input params to pc
p <- imputed_data[[1]] %>% length()

cat("Fitting with alpha =", alpha, "\n")
cat("######################################################################################\n")
cat("\n")
cat("MI\n")
tic()
start_timeMI <- Sys.time()
cuda_tPC_MI <- cu_pc_MI(suffStatMI, p = p, alpha = alpha, m.max = max_order, df_method = "old")
end_timeMI <- Sys.time()
timeMI <- as.numeric(difftime(end_timeMI, start_timeMI, units = "secs"))
print(cuda_tPC_MI@graph)
cat("\n")
cat("cuda_tPC_MI ord =", cuda_tPC_MI@max.ord, "\n")
cat("alpha    =", alpha, "\n\n")
cat("Time consumed:\n")
toc()
cat("\n")

cat("\n")
cat("SI\n")
tic()
start_timeSI <- Sys.time()
cuda_tPC_SI <- pc(suffStatMI, indepTest = micd::gaussMItest, p = p, alpha = alpha, m.max = max_order)
end_timeSI <- Sys.time()
timeSI <- as.numeric(difftime(end_timeSI, start_timeSI, units = "secs"))
print(cuda_tPC_SI@graph)
cat("\n")
cat("cuda_tPC_SI ord =", cuda_tPC_SI@max.ord, "\n")
cat("alpha    =", alpha, "\n\n")
cat("Time consumed:\n")
toc()
cat("\n")


# Calculate proportion of time
time_ratio <- timeSI / timeMI
cat("timeMI / timeSI =", time_ratio, "\n")

cat("shd:", pcalg::shd(cuda_tPC_SI, cuda_tPC_MI), "hd:", pcalg::shd(ugraph(cuda_tPC_SI@graph), ugraph(cuda_tPC_MI@graph)), "\n")
cat("######################################################################################\n\n\n")
