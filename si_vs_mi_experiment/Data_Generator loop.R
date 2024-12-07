#================================= Create DataSet =================================
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)
library(mice)
library(miceadds)
source("si_vs_mi_experiment/cohort_sim_utils.R")
# imputation/simulation part of experiment
# simulate data
# ampute 
# impute with m=1000
# choose m = 100 random imputations
# choose m = 10 random imputations
# choose m = 1 random imputation

# fitting experiment
# choose all imputations
# choose m = 100 random imputations
# choose m = 10 random imputations
# choose m = 1 random imputation
# get suffStats for all of these
# fit the (cu?)-tpc algorithm for all of these
# fit the complete-case PC-algorithm
# measure precision, recall, F1?, HD, SHD
# config 1,3000 --- done: NO
      # n = 3000,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.9,
      # ampute_proportion_tier = 0.1,
      # ampute_proportion_dropout = 0.1,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2
# config 1,100--- done: NO
      # n = 100,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.9,
      # ampute_proportion_tier = 0.1,
      # ampute_proportion_dropout = 0.1,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2
# config 2,3000 --- done: NO
      # n = 3000,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.9,
      # ampute_proportion_tier = 0,
      # ampute_proportion_dropout = 0,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2
# config 2,100 --- done: NO
      # n = 3000,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.9,
      # ampute_proportion_tier = 0,
      # ampute_proportion_dropout = 0,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2
# config 3,3000 --- done: NO
      # n = 3000,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.5,
      # ampute_proportion_tier = 0.1,
      # ampute_proportion_dropout = 0.1,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2
# config 3,100 --- done: NO
      # n = 100,
      # p = 33,
      # num_tiers = 5,
      # in_tier_prob = 0.1,
      # cross_tier_prob = 0.15,
      # lB = 0.1, 
      # uB = 1,
      # ampute_proportion_total = 0.5,
      # ampute_proportion_tier = 0,
      # ampute_proportion_dropout = 0,
      # n_dropout_tiers_to_be_excluded = 1,
      # n_amputation_dependencies = 2

config_name <- "config1n3000"
n_datasets <- 100
m <- 1000
for (i in 36:n_datasets) {
    print(i)
    tic()
    set.seed(i)
    # Generate a random DAG and synthetic data
    cohort <- simulate_ampute_cohort(
      n = 3000,
      p = 33,
      num_tiers = 5,
      in_tier_prob = 0.1,
      cross_tier_prob = 0.15,
      lB = 0.1, 
      uB = 1,
      ampute_proportion_total = 0.9,
      ampute_proportion_tier = 0.1,
      ampute_proportion_dropout = 0.1,
      n_dropout_tiers_to_be_excluded = 1,
      n_amputation_dependencies = 2
    )

    data <- cohort$data
    amputed_data <- cohort$amputed_data
    ordering <- cohort$tier_ord
    true_dag <- cohort$dag
    true_tmpdag <- cohort$tmpdag
    dataset_filename <- paste0("si_vs_mi_experiment/data/", config_name, "/dataset", i, ".csv")
    write.table(data, file = dataset_filename, row.names = FALSE, na = "", col.names = FALSE, sep = ",")

    dataset_path <- file.path(dataset_filename, fsep = .Platform$file.sep)
    imputed_data <- mice(amputed_data, m = m, method = "norm", printFlag = FALSE, remove.collinear = TRUE)

    setwd(paste0("/home/freddy-spaghetti/uni/tpc-cuda-MI/si_vs_mi_experiment/imputed_datasets_cohort_", config_name))
    # write mids obj
    #write.mice.imputation(imputed_data, paste0("imp", i), dattype = "csv", mids2spss = FALSE)
    save(imputed_data, file = paste0("imp", i, ".Rdata"))
    #write.table(imputed_data,file="data/dataset_imputed.csv", row.names=FALSE, na= "",col.names= FALSE, sep=",")
    setwd("/home/freddy-spaghetti/uni/tpc-cuda-MI")
    toc()
}
