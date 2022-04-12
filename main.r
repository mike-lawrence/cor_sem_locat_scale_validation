library(tidyverse)
`%<>%` = magrittr::`%<>%`
source('_imports.r')

#start parallel compile jobs in the background
compile_stan_files_as_parallel_jobs(path='stan')

nI = 10
reps = 10

(
	expand_grid(
		id = 1:nI
		, ab = factor(c('A','B'))
		, rep = 1:reps
	)
	%>% mutate(
		y = rnorm(n())
	)
) ->
	dat


(
	dat
	%>% select(-y,-rep)
	%>% distinct()
	%>% mutate(
		contrasts = get_contrast_matrix_rows_as_list(
			data = .
			, formula = ~ ab
			, contrast_kind = halfsum_contrasts
		)
	)
) ->
	Xc_with_vars

(
	Xc_with_vars
	%>% unnest(contrasts)
	%>% View()
)

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

	# Y: observations
	# vector[nY] Y ;
	, Y = dat$y

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

	# nY: num entries in the LA observation vector
	# int nY ;
	, nY = length(Y)

)


mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_gauss.stan')

# Sample the prior ----
data_for_stan$prior_informed = 1
data_for_stan$likelihood_informed = 0

prior_predictive_output = sample_mod(
	data = data_for_stan
	, mod = mod
	, max_treedepth = 11 # 10 is default
	, refresh_perc = 10
	, init = 2 # default of 2 is good for Gaussian likelihoods, lower may be necessary for binomial
)

#check diagnostics
prior_predictive_output %<>% add_draws_and_diagnostics_attr()
print(attr(prior_predictive_output,'dd')$sampler_diagnostics_across_chain_summary)
(
	attr(prior_predictive_output,'dd')$par_summary
	%>% select(rhat,contains('ess'))
	%>% summary()
)

# Generate yreps from a single prior draw ----
gq_mod = cmdstanr::cmdstan_model('stan/hierarchical_cor_plus_sem_locat_scale_gauss_GQ_yrep.stan')

(
	attr(prior_predictive_output,'dd')$draws
	%>% filter(.draw==1)
	%>% posterior::as_draws_array()
) ->
	prior_predictive_draw_for_yrep

prior_predictive_draw_gq = gq_mod$generate_quantities(
	data = data_for_stan
	, fitted_params = prior_predictive_draw_for_yrep
)

(
	prior_predictive_draw_gq$draws(format='draws_df')
	%>% as_tibble()
	%>% select(
		.chain,.iteration,.draw
		, contains('Y_rep')
	)
	%>% pivot_longer(
		cols = -c(.chain,.iteration,.draw)
		, names_to = 'variable'
	)
	%>% separate(
		variable
		, into=c('variable','index','dummy')
		, sep=c('[\\[\\]]')
		, fill = 'right'
		, convert = TRUE
	)
	%>% select(-dummy,-.chain,-.iteration,-.draw)
	%>% arrange(variable,index)
) ->
	prior_predictive_draw_yreps

(
	prior_predictive_draw_yreps
	%>% filter(variable=='Y_rep')
	%>% pull(value)
) ->
	data_for_stan$Y

# sample given the yrep of the prior draw ----
data_for_stan$likelihood_informed = 1

post = sample_mod(
	data = data_for_stan
	, mod = mod
	, max_treedepth = 11 # 10 is default
	, refresh_perc = 10
	, init = 2 # default of 2 is good for Gaussian likelihoods, lower may be necessary for binomial
)

#check diagnostics
post %<>% add_draws_and_diagnostics_attr()
print(attr(post,'dd')$sampler_diagnostics_across_chain_summary)
(
	attr(post,'dd')$par_summary
	%>% select(rhat,contains('ess'))
	%>% summary()
)

# plot_par(
# 	par=c('locat_intercept_mean','locat_coef_mean')
# 	, true = prior_predictive_draw_for_yrep
# )

plot_par(
	post
	, par_subst = '_mean'
	, true = prior_predictive_draw_for_yrep
)

plot_par(
	post
	, par_subst = '_sd'
	, true = prior_predictive_draw_for_yrep
)


# plot_par(
# 	par = 'locat_cors'
# 	, true = prior_predictive_draw_for_yrep
# )
plot_par(
	post
	, par_substr = 'cors'
	, true = prior_predictive_draw_for_yrep
)

