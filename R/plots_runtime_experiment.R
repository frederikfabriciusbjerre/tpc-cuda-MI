library(ggplot2)
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)
library(mice)
library(micd)
library(miceadds)
library(dplyr)
library(gridExtra)

results_cpu <- read.csv("m-p-alpha-experiment-cpu.csv")
results_gpu <- read.csv("m-p-alpha-experiment.csv")
# Calculate mean and standard deviation of runtimes
summary_results_cpu <- aggregate(run_time ~ p + m + alpha, data = results_cpu, FUN = function(x) c(mean = mean(x), sd = sd(x)))
summary_results_cpu <- do.call(data.frame, summary_results_cpu)
summary_results_gpu <- aggregate(run_time ~ p + m + alpha, data = results_gpu, FUN = function(x) c(mean = mean(x), sd = sd(x)))
summary_results_gpu <- do.call(data.frame, summary_results_gpu)
colnames(summary_results_cpu) <- c("p", "m", "alpha", "mean_runtime", "sd_runtime")
colnames(summary_results_gpu) <- c("p", "m", "alpha", "mean_runtime", "sd_runtime")

# Set y-axis limits based on the range of mean_runtime across both CPU and GPU data
y_limits <- range(c(log10(summary_results_cpu$mean_runtime), log10(summary_results_gpu$mean_runtime)), na.rm = TRUE)
# 1. Plot: Runtime vs log10(m) for each p and alpha
cpu_plot_m <- ggplot(summary_results_cpu, aes(x = log10(m), y = log10(mean_runtime), color = factor(p), shape = factor(alpha))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(p, alpha))) +
  labs(title = "Log runtime vs log number of imputations for pcalg", x = "log10(m)", y = "log10(Mean Runtime)", color = "p", shape = "alpha") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

gpu_plot_m <- ggplot(summary_results_gpu, aes(x = log10(m), y = log10(mean_runtime), color = factor(p), shape = factor(alpha))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(p, alpha))) +
  labs(title = "Log runtime vs log number of imputations for cuPC-MI", x = "log10(m)", y = "", color = "p", shape = "alpha") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

# 2. Plot: Runtime vs alpha for each p and m
cpu_plot_alpha <- ggplot(summary_results_cpu, aes(x = alpha, y = log10(mean_runtime), shape = factor(p), color = factor(m))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(p, m)), linetype = "solid") +
  labs(title = "Log runtime vs alpha for pcalg", x = "alpha", y = "log10(Mean Runtime)", color = "m", shape = "p") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

gpu_plot_alpha <- ggplot(summary_results_gpu, aes(x = alpha, y = log10(mean_runtime), shape = factor(p), color = factor(m))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(p, m)), linetype = "solid") +
  labs(title = "Log runtime vs alpha for cuPC-MI", x = "alpha", y = "", color = "m", shape = "p") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

# 3. Plot: Runtime vs p for each m and alpha
cpu_plot_p <- ggplot(summary_results_cpu, aes(x = p, y = log10(mean_runtime), shape = factor(m), color = factor(alpha))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(m, alpha)), linetype = "solid") +
  labs(title = "Log runtime vs p for pcalg", x = "p", y = "log10(Mean Runtime)", color = "m", shape = "alpha") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

gpu_plot_p <- ggplot(summary_results_gpu, aes(x = p, y = log10(mean_runtime), shape = factor(m), color = factor(alpha))) +
  geom_point(size = 2) +
  geom_line(aes(group = interaction(m, alpha)), linetype = "solid") +
  labs(title = "Log runtime vs p for cuPC-MI", x = "p", y = "", color = "m", shape = "alpha") +
  theme_minimal() +
  scale_shape_manual(values=c(15,16,17,18,4,3,7,8)) +
  ylim(y_limits)

# Arrange the plots side by side for each pair
grid.arrange(cpu_plot_m, gpu_plot_m, cpu_plot_alpha, gpu_plot_alpha, cpu_plot_p, gpu_plot_p, ncol = 2)
ggsave(filename="runtime_experiment.pdf", plot=grid.arrange(cpu_plot_m, gpu_plot_m, cpu_plot_alpha, gpu_plot_alpha, cpu_plot_p, gpu_plot_p, ncol = 2))
# Display a summary table
summary_results_cpu
summary_results_gpu

