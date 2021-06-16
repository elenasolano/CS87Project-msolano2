#!/usr/bin/awk -f

# CS87 - Final Project
# Maria-Elena Solano
#
# This awk script analyzes the results for the benchmark implementation of 
# pi-DFT on NVIDIA GPU.
# 
# To run, execute
#
#   awk -f analyze-results.awk \
#     fourier-parallel-pi-gpu-cuda-results.csv | sort -n -t 1;
#

{
  # Values required to display the average
  n_obs[$1] = $1;  
  p_obs[$2] = $2;  
  t_sum[$1,$2] += $3;
  t_obs_law[$1,$2] = $1*(($2-1)/$2) + ($1/$2)*(log($1/$2)/log(2));

  # Values required for the regression
  t[$0] = $3;  t_law[$0] = $1*(($2-1)/$2) + ($1/$2)*(log($1/$2)/log(2));
}
END{
  # Retrieve the number of replications
  K = NR/(length(n_obs)*length(p_obs));

  # Regress t against t_law, and determine the quality of the fit
  for(i in t){ At_A+=t_law[i]*t_law[i]; At_b+=t[i]*t_law[i]; }; 
  beta = At_b/At_A;
  N    = length(t);
  for(i in t){  t_m+=t[i]; t_law_m+=t_law[i];  };  t_m /= N;  t_law_m /= N;
  for(i in t){ 
    SS_res+=(t[i]     - beta*t_law[i])^2;
    SS_tot+=(t[i]     - t_law_m)^2;
    SS_prd+=(t_law[i] - l_law_m)^2;
  }
  R2      = 1 - SS_res/SS_tot;
  eta     = length(t) - 2;
  t_score = beta*sqrt(eta)/sqrt(SS_res/SS_prd);
  a = 1/sqrt(eta);
  if(eta % 2 == 0){ a /= 2; for(e=eta-1; e>2; e-=2){ a*=e/(e-1); }; }
  else{ a/=atan2(0,-1); for(e=eta-1; e>1; e-=2){ a*=e/(e-1); }; }  
  a /= sqrt(1+t_score^2/eta)^eta;

  # For the largest value of p, display the average, and the l2 fit.
  for(p in p_obs){ if(pi < p){ pi = p; }; };
  printf "Empirical time complexity of pi-DFT on NVIDIA GPU ";
  printf "(p=%d, %d replications):\n", pi, K;
  printf "Fit significant at the alpha < ";
  if(a < 1e-15){   # if overflow, use an estimate of the order of magnitude.
    a_exp = int(42.480516 - 1.216025*t_score - 1.366767*eta);
    if(a_exp < -120){ a_exp = -120; }
    printf "1.0e%d", a_exp;
  }
  else{ printf "%.2e", a; };
  printf " level.\n", a;
  printf "Input size (n)\tAvg. time (ms)\tTheta(n(p-1)/p + (n/p)log(n/p))\n";
  for(n in n_obs){
    printf "%u\t\t%.2f\t\t%.2f\n", n, t_sum[n,pi]/K, beta*t_obs_law[n,pi];
  }
}
