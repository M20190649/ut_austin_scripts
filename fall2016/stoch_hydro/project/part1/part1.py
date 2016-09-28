import statswrap

stats = statswrap.Statistics('nwm_med_onionck_091116.csv', 'flow_cfs', 'stats/')
# stats = statswrap.Statistics('rcurve_onionck.csv', 'dep', 'stats/')

# Problem 1a
sturges = stats.get_sturges()
stats.plot_hist('sturges')
stats.plot_hist(5)
stats.plot_hist(15)

# Problem 1b
stats.cumul_freq_dist()

# Problem 1c
stats.write_stats()

# Problem 1d
stats.boxplot()

# Problem 2
stats.norm_dist(0,1,-3,3,100)
stats.lognorm_dist(0,1,0,10,100)
stats.gamma_dist([0.9,2],0.5,0,30,30)