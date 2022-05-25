cat('sample_mod() now available as a function\n')
sample_mod = function(
	data
	, mod
	, preset = c('frugal','thorough')
	, ...
){
	if(	length(...names())>0){
		usr_dots = list(...)
		names(usr_dots) = ...names()
	}else{
		usr_dots = NULL
	}
	if(!(preset[1] %in% c('frugal','thorough'))){
		stop(paste0('"',preset[1],'" is not a valid preset. Valid presets include "frugal" & "thorough" only.'))
	}
	if(length(preset)>0){
		preset = preset[1]
		cat(crayon::cyan(paste0('Using preset "',preset,'"\n')))
	}
	sampling_config = lst(
		data = data
		, chains = parallel::detectCores()/2
		, parallel_chains = chains
		, iter_warmup = 1e3
		, iter_sampling = 1e3
		, percent = 10
		, max_treedepth = 10
	)
	if(preset=='thorough'){
		sampling_config %<>% (
			.
			%>% list_modify(
				max_treedepth = 12
				, metric = 'dense_e'
				, iter_warmup = 1e4
				, sig_figs = 18
				, save_warmup = T
				, save_latent_dynamics = T
			)
		)

	}
	#now update to account for dots
	if(!is.null(usr_dots)){
		sampling_config %<>% list_modify(usr_dots)
	}
	if(!('refresh'%in%names(sampling_config))){
		sampling_config$refresh = ifelse(
			sampling_config$percent==0
			, 1
			, ifelse(
				is.infinite(sampling_config$percent)
				, 0
				, round((sampling_config$iter_warmup+sampling_config$iter_sampling)*(sampling_config$percent/100))
			)
		)
	}
	sampling_config %<>% list_modify(percent=zap())
	if(!('output_dir'%in%names(sampling_config))){
		ramdisk_path = getOption('ramdisk_path')
		if(!is.null(ramdisk_path)){
			sampling_config$output_dir = ramdisk_path
		}else{
			sampling_config$output_dir = 'stan_sample_csvs'
		}
	}
	fs::dir_create(sampling_config$output_dir)
	post = do.call(mod$sample,sampling_config)
	attr(post,"sampling_config")=sampling_config
	return(post)
}
