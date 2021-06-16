#!/usr/bin/Rscript

# CS87 - Final Project
# Maria-Elena Solano
#
# This R script analyzes the results for pi-DFT on multicore CPU (pthreads).
# 
# To run, execute
#
#   Rscript --vanilla analyze-results.R;
#

# Function declarations
#
log2 = function(x){ log(x)/log(2); }

# Verify that the results file exists
if(!file.exists("fourier-parallel-pi-xeonphi-openmp-results.csv")){
  cat("Sorry, could not find fourier-parallel-pi-xeonphi-openmp-results.csv\n")
  quit();
}

# Read in the results
r = read.table(file   = "fourier-parallel-pi-xeonphi-openmp-results.csv", 
               header = FALSE, 
               sep    = "\t",
               col.names = c("n","p","time","time_tr","time_cy"));

# Determine the experimental parameters
n = sort(unique(r$n));  n_min = min(n);  n_max = max(n);
p = sort(unique(r$p));  p_min = min(p);  p_max = max(p);
K = nrow(r) / (length(n)*length(p));

# Compute the predicted law
time_tr_law = function(n,p){ n*(p-1)/p; };
time_cy_law = function(n,p){ (n/p)*log2(n/p); };
time_law    = function(n,p){ time_tr_law(n,p) + time_cy_law(n,p); };
r$time_tr_hat = time_tr_law(r$n, r$p);
r$time_cy_hat = time_cy_law(r$n, r$p);
r$time_hat    = time_law(r$n, r$p);
# Regress time against n((p-1)/p) + (n/p)log(n/p)
cat(sep="",
  "  Testing the hypothesis that the parallel time complexity follows ",
    "the law\n  [Funnel stage] + [Tube stage] = ",
    "Theta([n((p-1)/p)] + [(n/p)*log(n/p)])...\n");
lm_time = lm(time ~ time_tr_hat + time_cy_hat - 1, data=r);
alpha = max(coef(summary(lm_time))[4], 1e-120);       # Bekenstein bound!
cat(ifelse(alpha < 0.1, "    Yes: ", "    No: "));
cat(sep="", 
  "the time complexity of pi-DFT is Theta([n((p-1)/p)]+[(n/p)*log(n/p)])\n",
  "    (Fit significant at the alpha < ", alpha, " level.)\n");
# Regress tree-stage time against n((p-1)/p)
cat(sep="",
  "  Testing the hypothesis that the 1st (funnel) stage is ",
    "Theta(n((p-1)/p))...\n");
lm_time_tr = lm(time_tr ~ time_tr_hat - 1, data=r);
alpha_tr = max(coef(summary(lm_time_tr))[4], 1e-120);
cat(ifelse(alpha_tr < 0.1, "    Yes: ", "    No: "));
cat(sep="",
  "the time complexity of the 1st stage of pi-DFT is Theta(n((p-1)/p)).\n",
  "    (Fit significant at the alpha < ", alpha_tr, " level.)\n");
# Regress cylinder-stage time against (n/p)log(n/p)
cat(sep="",
  "  Testing the hypothesis that the 2nd (tube) stage is ", 
    "Theta((n/p)*log(n/p))...\n");
lm_time_cy = lm(time_cy ~ time_cy_hat - 1, data=r);
alpha_cy = max(coef(summary(lm_time_cy))[4], 1e-120);
cat(ifelse(alpha_cy < 0.1, "    Yes: ", "    No: "));
cat(sep="",
  "the time complexity of the 2nd stage of pi-DFT is Theta((n/p)*log(n/p))\n",
  "    (Fit significant at the alpha < ", alpha_cy, " level.)\n");

# Compute empirical speedup
empirical_speedup = function(K, n, p, r){
  s = c(NA);  length(s) = nrow(r);
  for(ni in n){
  	for(pi in p){
  	  s[r$p==pi&r$n==ni] = r$time[r$p==min(p)&r$n==ni]/r$time[r$p==pi&r$n==ni];
  	}
  }
  return(s);
}
r$speedup = empirical_speedup(K, n, p, r);

# Compute fitted speedup
fitted_speedup = function(ni, p, lm_time){
  time_tr_hat  = time_tr_law(ni, p);
  time_cy_hat  = time_cy_law(ni, p);
  time_pred = predict(lm_time, newdata=data.frame(time_tr_hat, time_cy_hat));
  return(time_pred[1]/time_pred);
}

# Compute fitted and average time
fitted_time_tr = function(ni, p, lm_time_tr){
  time_tr_hat  = time_tr_law(ni, p);
  time_tr_pred = predict(lm_time_tr, newdata=data.frame(time_tr_hat));
  return(time_tr_pred);
}
fitted_time_cy = function(ni, p, lm_time_cy){
  time_cy_hat  = time_cy_law(ni, p);
  time_cy_pred = predict(lm_time_cy, newdata=data.frame(time_cy_hat));
  return(time_cy_pred);
}
avg_time_tr = function(ni, p, r){
  t = c(NA);  length(t) = length(p);
  for(i in 1:length(p)){
    t[i] = mean(r[r$n==ni & r$p==p[i],]$time_tr);
  }
  return(t);
}
avg_time_cy = function(ni, p, r){
  t = c(NA);  length(t) = length(p);
  for(i in 1:length(p)){
    t[i] = mean(r[r$n==ni & r$p==p[i],]$time_cy);
  }
  return(t);
}

# And plot the results, including the fitted law
# Note: just plotting for the min, mid and max n (for brevity)
n_plot = c(min(n), n[floor(length(n)/2)], max(n));
for(ni in n_plot){
  invisible(pdf(width=14, height=7, file=sprintf(
    "fourier-parallel-pi-xeonphi-openmp-results-analysis-n%d.pdf", ni)));
  r_sub = r[r$n==ni,];

  par(mfrow=c(1,2));

  # First plot
  plot(0, pch='', 
    main=sprintf(paste(sep="",
    "Empirical speedup of pi-DFT on Xeon Phi (%d replications)\n",
    "and n = %d. Fit significant at the alpha < %.2e level"), K, ni, alpha), 
  xlim=c(min(p)-0.05,max(p)+0.05), xlab="Number of processors (p)", xaxt="n",
  ylim=c(min(p)-0.05,max(p)+0.05), ylab="Speedup", cex.axis=0.8);
  axis(side=1, at=p, cex.axis=0.8);
  points(r_sub$p, r_sub$speedup, pch=4, col=rgb(0.3,0.3,0.3,0.2));
  lines(p, fitted_speedup(ni, p, lm_time),
    lwd=2, col=rgb(0.2,0.5,0.2,0.8));
  
  # Second plot
  barplot(col=c("#629627", "#623034"),
    height=rbind(avg_time_tr(ni, p, r),
                 avg_time_cy(ni, p, r)),
    main=sprintf(paste(sep="",
    "Average time of pi-DFT on Xeon Phi (%d replications)\n",
    "and n = %d (tree stage in green)"), K, ni, alpha), 
          names.arg=p,
          xlab="Number of processors (p)",
          ylab="Average time (ms)");
}

# Done!
cat(sep="",
  "Done. See also:\n", paste(
  "  fourier-parallel-pi-xeonphi-openmp-results-analysis-n",n_plot,".pdf\n",
  sep="", collapse=""));
