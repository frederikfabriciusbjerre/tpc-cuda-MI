library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)
library(mice)
library(micd)
library(miceadds)
library(dplyr)
library(bnlearn)
source("R/MeeksRules.R")
source("R/tcheckTriple.R")
source("R/tpc_cons_intern.R")
source("R/tskeleton.R")
source("R/tskeleton_parallel.R")
source("R/tpc.R")
source("R/fixedMItest.R")
source("cuda/cuPCMI.R")
library(ggplot2)

# List of networks and their names
network_list <- list(
  ecoli =       readRDS("bnlearn_networks/ecoli70.rds"),
  magic_irri =  readRDS("bnlearn_networks/magic-irri.rds"),
  magic_niab =  readRDS("bnlearn_networks/magic-niab.rds"),
  arth150 =     readRDS("bnlearn_networks/arth150.rds")
)

# Initialize an empty data frame to store results
results <- data.frame()

# Set alpha value for the PC algorithm
alpha <- 0.05

# Define the degrees of freedom correction methods
df_correction_methods <- c('old', 'br', 'reiter') 

i <- 0
# Start the experiment
for (network_name in names(network_list)) {
  # Load the network
  network <- network_list[[network_name]]
  
  cat("Processing network:", network_name, "\n")
  
  # Get the number of nodes
  p <- length(nodes(network))
  
  # Convert the true network to a graphNEL object
  dag_graphNEL <- as.graphNEL(network)
  
  # Get the adjacency matrix of the true DAG
  dag_adjmat <- as(dag_graphNEL, "matrix")
  
  for (m in c(100, 10, 5)) {
    for (n in c(1000, 25)) {
        for (prop_miss in c(0.1, 0.5, 0.9)){
            for (epoch in 1:10) {
                i <- i + 1
                cat("Network:", network_name, ", n =", n, ", m =", m, ", epoch =", epoch, "\n")
                
                # Generate synthetic data from the network
                set.seed(i)  # For reproducibility
                data_sample <- rbn(network, n = n)
                
                # Compute the true CPDAG
                true_cpdag <- cpdag(network)
                
                cat("imputing... \n")
                # Introduce missing data
                data_missing <- mice::ampute(data_sample, prop = prop_miss, 
                                            mech = "MAR", 
                                            bycases = TRUE)$amp
                
                # Multiple imputations
                tryCatch({
                    imputed_data <- mice::mice(data_missing, m = m, method = "norm",
                                        visitSequence = "monotone", printFlag = FALSE)
                }, error = function(e) {
                    # Message to indicate that an error occurred
                    message(paste("Error in iteration", i, ":", e$message))
                    # Skipping this iteration
                })
                
                # Get sufficient statistics for imputed data
                suff_imputed <- micd::getSuff(imputed_data, test = "gaussMItest")
                
                cat("pcalg fitting... \n")
                # Run PC algorithm on complete data (without missing values)
                suff_complete <- list(C = cor(data_sample), n = nrow(data_sample))
                pc_complete <- pcalg::pc(suffStat = suff_complete, indepTest = pcalg::gaussCItest, 
                                        alpha = alpha, p = p, m.max = 12)
                adjmat_complete <- as(pc_complete@graph, "matrix")
                
                # Run PC algorithm on imputed data with each df_correction_method
                
                for (df_correction_method in df_correction_methods) {
                    cat("cuPC fitting with", df_correction_method, "... \n")
                    pc_imputed <- cu_pc_MI(suff_imputed, p = p, alpha = alpha, m.max = 12, df_method = df_correction_method)
                    
                    adjmat_imputed <- as(pc_imputed@graph, "matrix")
                    
                    # Get the skeletons (convert to undirected graphs)
                    adjmat_imputed_skeleton <- (adjmat_imputed + t(adjmat_imputed)) > 0
                    
                    # Flatten adjacency matrices
                    adj_imputed_vector <- as.vector(adjmat_imputed_skeleton)
                    adjmat_complete_vector <- as.vector(adjmat_complete)
                    
                    # Compute TP, FP, FN, TN
                    TP <- sum(adj_imputed_vector == 1 & adjmat_complete_vector == 1)
                    FP <- sum(adj_imputed_vector == 1 & adjmat_complete_vector == 0)
                    FN <- sum(adj_imputed_vector == 0 & adjmat_complete_vector == 1)
                    TN <- sum(adj_imputed_vector == 0 & adjmat_complete_vector == 0)
                    
                    recall <- if ((TP + FN) > 0) TP / (TP + FN) else NA
                    precision <- if ((TP + FP) > 0) TP / (TP + FP) else NA
                    
                    # Compute Structural Hamming Distance (SHD)
                    shd_value <- pcalg::shd(pc_imputed@graph, pc_complete@graph)
                    
                    # Hamming distance between skeletons
                    hamming_distance <- pcalg::shd(ugraph(pc_imputed@graph), ugraph(pc_complete@graph))
                    
                    # Store results
                    results <- rbind(results, data.frame(
                        network = network_name,
                        p = p,
                        m = m,
                        n = n,
                        epoch = epoch,
                        df_correction_method = df_correction_method,
                        prop_miss = prop_miss, 
                        recall = recall,
                        precision = precision,
                        hamming_distance = hamming_distance,
                        shd = shd_value,
                        stringsAsFactors = FALSE
                    ))
                    write.table(
                        results[nrow(results), ], 
                        file = "results.csv", 
                        sep = ",", 
                        row.names = FALSE, 
                        col.names = !file.exists("results.csv"),  # Write header only if file doesn't exist
                        append = TRUE
                    )
                    }
                }
            }
        }
    }
}

# Display the results
print(results)
