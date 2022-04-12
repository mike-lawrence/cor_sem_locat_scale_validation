cat('plot_coefs() now available as a function\n')
plot_par = function(post,par=NULL,par_substr=NULL,title=NULL,trim_p=NULL,true=NULL){
	`%<>%` <- magrittr::`%<>%`
	to_plot = extract_pars(post,par)
	if(!is.null(par_substr)){
		to_plot %<>% filter(str_detect(variable,par_substr))
	}
	if(!is.null(trim_p)){
		(
			to_plot
			%>% select(variable,data)
			%>% unnest(data)
			%>% group_by(variable)
			%>% summarise(
				data = list(tibble(value=trim_tails(value,trim_p)))
				, .groups = 'drop'
			)
			%>% right_join(
				(to_plot %>% select(-data))
				, by = c('variable')
			)
		) ->
			to_plot
	}
	(
		to_plot
		%>% ggplot()
		+ geom_hline(yintercept = 0,linetype=3,colour='white')
		+ geom_violin(
			data = (
				.
				%>% select(variable,data)
				%>% unnest(data)
			)
			, mapping = aes(
				x = variable
				, y = value
			)
			, colour = 'transparent'
			, fill = 'black'
			, scale = 'width'
			, alpha = .5
		)
		+ geom_linerange(
			data = (
				.
				%>% select(variable,data)
				%>% unnest(data)
				%>% group_by(variable)
				%>% summarise(
					q0 = min(value)
					, q100 = max(value)
					, .groups = 'drop'
				)
			)
			, mapping = aes(
				x = variable
				, ymin = q0
				, ymax = q100
			)
			, colour = 'black'
			# , alpha = .5
			, size = .5
		)
		+ geom_linerange(
			mapping = aes(
				x = variable
				, ymin = q6.5
				, ymax = q93.5
				, colour = ess_tail_is_bad
			)
		)
		+ geom_linerange(
			mapping = aes(
				x = variable
				, ymin = q25
				, ymax = q75
				, colour = ess_bulk_is_bad
			)
			, size = 3
		)
		+ geom_point(
			mapping = aes(
				x = variable
				, y = q50
				, fill = rhat_is_bad
			)
			, shape = 21
			, size = 2
			, colour = 'black'
		)
		+ geom_hline(yintercept=0,linetype=3)
		+ coord_flip()
		+ scale_color_manual(
			values = lst(`TRUE`='red',`FALSE`='white')
			, labels = lst(`TRUE`='<100',`FALSE`='>=100')
		)
		+ scale_fill_manual(
			values = lst(`TRUE`='red',`FALSE`='black')
			, labels = lst(`TRUE`='>1.01',`FALSE`='<=1.01')
		)
		+ labs(
			y = "Posterior Value of Parameter"
			, x = 'Parameter'
			, colour = 'ESS'
			, fill = 'Rhat'
			, title = title
		)
		+ theme(
			aspect.ratio = 2/(1+sqrt(5))
		)
	) ->
		p
	if(!is.null(true)){
		(
			p
			+ geom_point(
				data = (
					true
					%>% posterior::as_draws_df()
					%>% pivot_longer(
						everything()
						, names_to = 'variable'
					)
					%>% filter( variable %in% unique(to_plot$variable) )
				)
				,  aes(
					x = variable
					, y  = value
				)
				, shape = 4
				, size = 3
				, colour = 'green'
			)
		) ->
			p
	}
	return(p)
}
