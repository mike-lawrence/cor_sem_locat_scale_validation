cat('compile_stan_files_as_parallel_jobs() now available as a function\n')
compile_stan_files_as_parallel_jobs = function(files=NULL,path=NULL){
	if(is.null(files)&is.null(path)){
		stop('Either "files" or "path" must be non-NULL')
	}
	if((!is.null(files))&(!is.null(path))){
		stop('One of "files" or "path" must be NULL')
	}
	if(!is.null(path)){
		files = fs::dir_ls(path=path,glob='*.stan')
	}
	for(file in files){
		mod_name = fs::path_ext_remove(fs::path_file(file))
		temp_file = tempfile()
		job_code = sprintf("
			file = '%s'
			cat('Compiling',file,'...')
			temp = NULL
			temp = cmdstanr::cmdstan_model(file)
			if(is.null(temp)){
				cat('Compiling',file,'failed.')
				stop()
			}else{
				cat('Compiling',file,'succeeded.')
			}
			"
			, as.character(file)
		)
		write(job_code,file=temp_file)
		rstudioapi::jobRunScript(
			path = temp_file
			, name = paste('Compiling',mod_name)
			, workingDir = getwd()
		)
	}
	return(invisible(NULL))
}
