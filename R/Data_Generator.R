#================================= Create DataSet =================================
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)
library(mice)
library(miceadds)

p <- 40 #number of nodes
probability <- 0.075
n <- 1000 #number of sample
vars <- c(paste0(1:p))

# mice params
prob_miss <- 0.1
m <- 10
method <- "norm"

set.seed(43)

gGtrue <- randomDAG(p, prob = probability, lB = 0.1, uB = 1, V = vars)
N1 <- runif(p, 0.5, 1.0)
Sigma1 <- matrix(0, p, p)
diag(Sigma1) <- N1
eMat <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma1)
gmG <- list(x = rmvDAG(n, gGtrue, errMat = eMat), g = gGtrue)
write.table(gmG$x,file="data/dataset.csv", row.names=FALSE, na= "",col.names= FALSE, sep=",")

dataset_path <- file.path("data/dataset.csv", fsep = .Platform$file.sep)
data <- read.table(dataset_path, sep = ",")


# missing at random data 
data_missing <- ampute(data, prop = prob_miss, 
                        mech = "MAR", 
                        bycases = TRUE)$amp

# naive mice imputation
imputed_data <- mice(data_missing, m = m, method = method, printFlag = FALSE, remove.collinear = TRUE)

system("rm -rf dataset_imputed")
# write mids obj
write.mice.imputation(imputed_data, "dataset_imputed", dattype = "csv", mids2spss = FALSE)
#write.table(imputed_data,file="data/dataset_imputed.csv", row.names=FALSE, na= "",col.names= FALSE, sep=",")
