# preamble ----

# install dependencies & source helpers
source('_imports.r')

# load tidyverse and a magrittr pipe
library(tidyverse)
`%>%` = magrittr::pipe_eager_lexical
`%<>%` = magrittr::`%<>%`

#make RStudio use cmdstan for syntax checking
enable_rstudio_stan_syntax_check()

#start parallel compile jobs in the background
compile_stan_files_as_parallel_jobs(path='stan')

#start a ramdisk (Mac/linux only)
# Note: will ask for sudo password, but it's not permanently; see `open(r_helpers/start_ramdisk.r)`
# Also note: ramdisk may even be *slower*; still working out why...
#start_ramdisk(gb=10)


# generate the data structure ----

#nI=60 & reps=10 takes about 5mins to sample
nI = 60
reps = 10

(
	expand_grid(
		id = 1:nI
		, ab = factor(c('A','B'))
		, cd = factor(c('C','D'))
		, rep = 1:reps
	)
	%>% mutate(
		y_gauss = 0 #just a dummy value for now; we'll generate data using Stan
		, y_binom = 0 #ditto
	)
) ->
	dat

# add contrasts
(
	dat
	%>% select(-contains('_y_'),-rep)
	%>% distinct()
	%>% mutate(
		contrasts = get_contrast_matrix_rows_as_list(
			data = .
			, formula = ~ ab*cd
			, contrast_kind = halfsum_contrasts
		)
	)
) ->
	Xc_with_vars

#quick view of the contrasts
(
	Xc_with_vars
	%>% unnest(contrasts)
	%>% select(-id)
	%>% distinct()
	%>% View()
)

#package the data for Stan
data_for_stan = lst(

	# Xc: condition-level predictor matrix
	# matrix[rXc,nXc] Xc ;
	Xc = (
		Xc_with_vars
		%>% select(contrasts)
		%>% unnest(contrasts)
		%>% as.matrix()
	)

	# iXc: which individual is associated with each row in Xc
	# array[rXc] int<lower=1,upper=nI> iXc ;
	, iXc = (
		Xc_with_vars
		%>% pull(id)
		%>% as.factor()
		%>% as.numeric()
	)

	# Y_gauss: observations modelled with location-scale Gaussian model
	# matrix[nY,2] Y_gauss ;
	, Y_gauss = (
		dat
		%>% pull(y_gauss)
	)

	# Y_binom: observations modelled with binomial model
	# array[nY,2] int<lower=0,upper=1> Y_binom ;
	, Y_binom = (
		dat
		%>% pull(y_binom)
	)

	# yXc: which row in Xc is associated with each observation in Y
	# array[nY] int<lower=1,upper=rXc> yXc ;
	, yXc = (
		Xc_with_vars
		%>% mutate(Xc_row = 1:n())
		# right-join with dat to preserve dat's row order
		%>% right_join((
			dat
			%>% mutate(
				dat_row = 1:n()
			)
		))
		%>% arrange(dat_row)
		%>% pull(Xc_row)
	)


	# nI: number of individuals
	# int<lower=2> nI ;
	, nI = length(unique(dat$id))

	# nXc: number of condition-level predictors
	# int<lower=2> nXc ;
	, nXc = ncol(Xc)

	# rXc: number of rows in the condition-level predictor matrix Xc
	# int<lower=nXc> rXc ;
	, rXc = nrow(Xc)

	# nY: num entries in the observation vector
	# int nY ;
	, nY = length(Y_gauss)

)


# Sample the prior (the easy way, using explicit rng functions in GQ) ----
prior_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_binom_GQ_prior.stan')
prior_post = prior_mod$generate_quantities(
	data = data_for_stan
	, fitted_params = posterior::draws_array(dummy=0)
)

# Sample the prior (the hard/thorough way that checks that the sampler can sample the prior implicitly without pathology) ----
sampling_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_binom.stan')
prior_post = sample_mod(
	# data = data_for_stan
	data = (
		data_for_stan
		%>% list_modify(
			prior_informed = 1
			, likelihood_informed = 0
		)
	)
	, mod = sampling_mod
	, preset = 'frugal' #options: frugal, thorough; former being cmdstanr defaults, latter using `iter_warmup=1e4,max_treedepth=12,metric='dense_e'`
	, precent = 10 # `percent=1`==`refresh=1``; `percent=Inf`==`refresh=0`
	# , ... #can supply any deviations-from-preset values here for mod$sample(...) arguments
)
#check diagnostics
prior_post %<>% add_draws_and_diagnostics_attr()
print(attr(prior_post,'dd')$sampler_diagnostics_across_chain_summary)
(
	attr(prior_post,'dd')$par_summary
	%>% select(rhat,contains('ess'))
	%>% summary()
)

# extract a single prior draw for predictive GQ ----
(
	prior_post$draws(inc_warmup = F)
	%>% posterior::subset_draws(draw=1)
) ->
	prior_post_draw_for_predictive

# tweak the reliabilities to be moderate-high:
prior_post_draw_for_predictive %<>% (
	.
	%>% posterior::as_draws_list()
	%>% pluck(1)
	%>% {function(x){
		for(i in 1:data_for_stan$nXc){
			x %<>% assign_in(paste0('locat_cors[',i,']'),.2)
		}
		return(x)
	}}()
	%>% posterior::as_draws_array()
)



# Generate yreps from the prior draw & assign into data_for_stan ----
predictive_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_binom_GQ_yrep.stan')
data_for_stan %<>% (
	.
	%>% predictive_mod$generate_quantities(
		fitted_params = prior_post_draw_for_predictive
	)
	%>% {function(x){
		x$draws(variables = c('Y_gauss_rep','Y_binom_rep'))
	}}()
	%>% posterior::as_draws_rvars()
	%>% {function(x){(
		data_for_stan
		%>% list_modify(
			Y_gauss = posterior::E(x$Y_gauss_rep)
			, Y_binom = posterior::E(x$Y_binom_rep)
		)
	)}}()
)


# sample given the prior & prior-predictive data ----
sampling_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_binom.stan')
post = sample_mod(
	data = (
		data_for_stan
		%>% list_modify(
			prior_informed = 1
			, likelihood_informed = 1
		)
	)
	, mod = sampling_mod
	, preset = 'frugal' #options: frugal, thorough; former being cmdstanr defaults, latter using `iter_warmup=1e4,max_treedepth=12,metric='dense_e'`
	, precent = 10 # `percent=1`==`refresh=1``; `percent=Inf`==`refresh=0`
	# , ... #can supply any deviations-from-preset values here for mod$sample(...) arguments
)

#check diagnostics
post %<>% add_draws_and_diagnostics_attr()
print(attr(post,'dd')$sampler_diagnostics_across_chain_summary)
(
	attr(post,'dd')$par_summary
	%>% select(rhat,contains('ess'))
	%>% summary()
)

plot_par(
	post
	, par_subst = '_mean'
	, true = prior_post_draw_for_predictive
)

plot_par(
	post
	, par_subst = '_sd'
	, true = prior_post_draw_for_predictive
)

plot_par(
	post
	, par_substr = 'cors'
	, true = prior_post_draw_for_predictive
)


# posterior predictive check ----
predictive_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_binom_GQ_yrep.stan')

post_predictive = predictive_mod$generate_quantities(
	data = data_for_stan
	, fitted_params = post$draws()
)

(
	c('Y_gauss_rep','Y_binom_rep')
	%>% post_predictive$draws()
	%>% posterior::as_draws_rvars()
) ->
	post_predictive_rep_draws

#todo: finish posterior predictive check
