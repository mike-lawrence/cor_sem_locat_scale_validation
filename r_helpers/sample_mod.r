cat('sample_mod() now available as a function\n')
sample_mod = function(
	data = data_for_stan
	, mod
	, chains = parallel::detectCores()/2
	, max_treedepth = 10
	, metric = 'dense_e' # 'diag_e' is default; 'dense_e' is slower but yields better sampling of a wider variety of posterior geometries
	, iter_warmup = 1e4 #1e4 gives greater-than-default confidence in warmup sufficiency
	, iter_sampling = 1e3 # 1e3 is nearly always a good value
	, refresh_perc = 10
	, init = 2 # default of 2 is good for Gaussian likelihoods, lower may be necessary for binomial
	# values derived from the above
	, parallel_chains = chains
	, refresh = round((iter_warmup+iter_sampling)/refresh_perc)
){
	sampling_config = lst(
		data = data_for_stan
		, mod = mod
		, chains = chains
		, max_treedepth = max_treedepth
		, metric = metric
		, iter_warmup = iter_warmup
		, iter_sampling = iter_sampling
		, refresh_perc = refresh_perc
		, init = init
		, draws_sampling = iter_sampling*chains
		, parallel_chains = parallel_chains
		, refresh = refresh
	)
	post = mod$sample(
		data = sampling_config$data
		, chains = sampling_config$chains
		, parallel_chains = sampling_config$parallel_chains
		, max_treedepth = sampling_config$max_treedepth
		, metric = sampling_config$metric
		, iter_warmup = sampling_config$iter_warmup
		, iter_sampling = sampling_config$iter_sampling
		, refresh = sampling_config$refresh
		, init = sampling_config$init
	)
	attr(post,"sampling_config")=sampling_config
	return(post)
}
