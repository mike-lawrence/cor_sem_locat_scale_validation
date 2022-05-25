cat('extract_pars() now available as a function\n')
extract_pars = function(post,par=NULL){
	`%<>%` <- magrittr::`%<>%`
	if(is.null(attr(post,'dd'))){
		stop('Please run `add_draws_and_diagnostics_attr()` first')
	}
	(
		post$draws(par)
		%>% posterior::as_draws_df()
		%>% as_tibble()
		%>% pivot_longer(
			cols = -c(.chain,.iteration,.draw)
			, names_to = 'variable'
		)
		%>% group_nest(variable)
		%>% filter(!str_detect(variable,'cholfaccorr'))
		%>% left_join(attr(post,'dd')$par_summary,by='variable')
		%>% mutate(
			rhat_is_bad = 1.01<rhat
			, ess_bulk_is_bad = 100>ess_bulk
			, ess_tail_is_bad = 100>ess_tail
		)
	) ->
		extracted
	return(extracted)
}
