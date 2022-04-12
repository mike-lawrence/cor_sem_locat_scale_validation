cat('add_draws_and_diagnostics() now available as a function\n')
add_draws_and_diagnostics_attr = function(post,quantile_probs=c(0,.065,.25,.5,.75,.935,1)){
	dd = list()
	(
		post$sampler_diagnostics()
		%>% posterior::as_draws_df()
		%>% {function(x){
			class(x) <- setdiff(class(x), c("draws_df", "draws"))
			return(x)
		}}()
	) ->
		dd$sampler_diagnostics

	# Check treedepth, divergences, & rebfmi
	(
		dd$sampler_diagnostics
		%>% group_by(.chain)
		%>% summarise(
			`% max treedepth hit` = 100*mean(treedepth__>attr(post,'sampling_config')$max_treedepth)
			, `% divergent` = 100*mean(divergent__)
			, `1/ebfmi` = var(energy__)/(sum(diff(energy__)^2)/n()) # n.b. reciprocal of typical EBFMI, so bigger=bad, like rhat
			, .groups = 'drop'
		)
	) ->
		dd$sampler_diagnostics_by_chain_summary

	(
		dd$sampler_diagnostics_by_chain_summary
		%>% summarise(
			any_max_treedepth_hit = any(`% max treedepth hit`>0)
			, any_divergent = any(`% divergent`>0)
			, `worst 1/ebfmi` = max(`1/ebfmi`)
		)
	) ->
		dd$sampler_diagnostics_across_chain_summary

	# `sampler_diagnostics_across_chain_max` is useful as target for checks in
	# automated workflows involving, for example, increasing sampling_config$max_treedepth
	# if any chains have >X% iterations hitting it, changing from "diag_e" to "dense_e", etc.


	(
		dd$sampler_diagnostics
		%>% as_tibble()
		%>% select(.chain,.iteration,treedepth__)
		%>% mutate(treedepth__=factor(treedepth__,levels=c(1:attr(post,'sampling_config')$max_treedepth)))
		%>% group_by(.chain)
		%>% count(treedepth__,.drop=F)
		%>% pivot_wider(names_from=.chain,values_from=n)
	) ->
		dd$treedepth_table

	# gather summary for core parameters (inc. r̂ & ess)
	(
		post$draws(format='draws_df')
		%>% as_tibble()
		%>% {function(x){
			class(x) <- setdiff(class(x), c("draws_df", "draws"))
			return(x)
		}}()
	) ->
		dd$draws

	(
		dd$draws
		%>% select(-contains('cholfaccorr')) #remove cholesky, as we have the lower tri too
		%>% posterior::summarise_draws(
			~ posterior::quantile2(.x, probs = quantile_probs)
			, posterior::default_convergence_measures()
			, .cores = parallel::detectCores()
		)
		%>% mutate(
			`% ess_bulk` = ess_bulk/attr(post,'sampling_config')$draws_sampling*100
			, `% ess_tail` = ess_tail/attr(post,'sampling_config')$draws_sampling*100
		)
	) ->
		dd$par_summary

	# View those with suspect r̂
	(
		dd$par_summary
		%>% filter(rhat>1.01)
	) ->
		suspects
	if(nrow(suspects)>0){
		cat('Some parameters have rhat>1.01; launching View thereof...')
		View(suspects)
	}

	attr(post,'dd') = dd
	return(post)

}
