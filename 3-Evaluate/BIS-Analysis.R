"""
BIS Action Comparison Analysis for Medical Treatment Optimization
=================================================================

This script performs comparative analysis of BIS (Bispectral Index) actions 
across different patient states using parallel computing for efficient processing 
of large-scale medical datasets from eICU database.
"""

# =========================
# 1. Environment Setup
# =========================

# Clear workspace and load required packages
rm(list = ls())

# Load parallel computing packages
library(doParallel)
library(foreach)
library(dplyr)

print("▌Initializing BIS action comparison analysis...")

# =========================
# 2. Data Loading and Preparation
# =========================

# Load eICU dataset (interactive file selection)
z <- read.csv(file.choose())

# Display dataset structure
str(z)

# Initialize bstate column for tracking patient states
z$bstate <- as.numeric(NA)

print("✔ Dataset loaded successfully")

# =========================
# 3. Parallel Data Processing
# =========================

# Split data by patient ID for parallel processing
lst <- split(z, z$id)
j <- as.numeric(length(lst))

print(paste("Processing", j, "patients in parallel..."))

# Set up parallel computing environment
num_cores <- 20  # Utilize 20 CPU cores for parallel processing
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Parallel processing of each patient's data
all_results <- foreach(m = 1:j, .combine = rbind, .packages = c()) %dopar% {
    tmp <- lst[[m]]  # Get current patient's data
    nrow <- as.numeric(nrow(tmp))   # Number of observations for current patient
    tp <- as.data.frame(NULL)  # Initialize results data frame
    
    # Process patient data in 300-observation chunks
    if (nrow > 300) {
        while (as.numeric(nrow(tmp)) > 300) {
            # Extract 300-observation window
            p <- tmp[1:300, ]
            # Set bstate to the state value at position 300
            p$bstate <- p[300, 3]
            tp <- rbind.data.frame(tp, p)
            
            # Update state values for remaining data
            if ("state" %in% colnames(tmp)) {
                tmp$state <- tmp$state - 1
            }
            tmp <- tmp[-1, ]  # Remove processed observation
        }
    }
    
    # Add remaining observations
    tp <- rbind.data.frame(tp, tmp)
    return(tp)
}

# Stop parallel cluster
stopCluster(cl)

print("✔ Parallel data processing completed")

# Display processed data structure
str(all_results)

# =========================
# 4. Save Processed Data
# =========================

# Save comprehensive results to file
write.csv(all_results, file = "/home/szus/ct.csv", row.names = FALSE, quote = FALSE)
print("✔ Processed data saved to /home/szus/ct.csv")

# =========================
# 5. Statistical Analysis Preparation
# =========================

# Create analysis subset (first 300,000 observations for efficiency)
t <- all_results[1:300000, ]

# Filter data for BIS states between 30-80 (clinically relevant range)
filtered_data <- t[all_results$bstate >= 30 & all_results$bstate <= 80, ]

print(paste("Analyzing", nrow(filtered_data), "observations in BIS range 30-80"))

# =========================
# 6. Parallel Statistical Computation
# =========================

# Define BIS state values for analysis
bstate_values <- 30:80

# Set up second parallel computing environment for statistical analysis
cl2 <- makeCluster(num_cores)
registerDoParallel(cl2)

print("▌Computing statistical metrics across BIS states...")

# Parallel computation of statistical metrics for each BIS state
stats_results <- foreach(bs = bstate_values, .combine = rbind, 
                         .packages = c("dplyr"), .export = "filtered_data") %dopar% {
                           
    # Filter data for current BIS state
    current_data <- filtered_data[filtered_data$bstate == bs, ]
    
    # Calculate comprehensive statistical metrics
    result <- data.frame(
        bstate = bs,
        action_mean = mean(current_data$action, na.rm = TRUE),
        action_sd = sd(current_data$action, na.rm = TRUE),
        val_mean = mean(current_data$val, na.rm = TRUE),
        val_sd = sd(current_data$val, na.rm = TRUE),
        diff_mean = mean(current_data$val - current_data$action, na.rm = TRUE),
        diff_sd = sd(current_data$val - current_data$action, na.rm = TRUE),
        n_observations = nrow(current_data)
    )
    
    return(result)
}

# Stop second parallel cluster
stopCluster(cl2)

print("✔ Statistical analysis completed")

# =========================
# 7. Results Examination and Export
# =========================

# Display statistical results
print(stats_results)

# Save statistical results to file
write.csv(stats_results, file = "/home/szus/stats_results.csv", 
          row.names = FALSE, quote = FALSE)

print("✔ Statistical results saved to /home/szus/stats_results.csv")
print("Analysis process completed successfully")

# =========================
# 8. Optional: Summary Statistics
# =========================

# Generate summary of the analysis
cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Total patients processed:", j, "\n")
cat("BIS states analyzed:", min(bstate_values), "-", max(bstate_values), "\n")
cat("Total observations in analysis:", nrow(filtered_data), "\n")
cat("Average observations per BIS state:", 
    round(mean(stats_results$n_observations), 1), "\n")

# Display key findings
if (nrow(stats_results) > 0) {
    max_diff_state <- stats_results[which.max(abs(stats_results$diff_mean)), "bstate"]
    cat("BIS state with largest action-value difference:", max_diff_state, "\n")
}
