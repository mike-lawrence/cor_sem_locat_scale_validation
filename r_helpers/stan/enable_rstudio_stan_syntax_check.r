cat('enable_rstudio_stan_syntax_check() now available as a function\n')
enable_rstudio_stan_syntax_check = function(){
	cat('RStudio "Check syntax on save" will now use latest cmdstan syntax checker')
	#dependencies: cmdstanr, fs, stringr, crayon
	cmdstan_pedantic_check = function(path){
		stdout = processx::run(
			command = fs::path(cmdstanr::cmdstan_path(),cmdstanr:::stanc_cmd())
			, args = c(
				basename(path)
				, '--include-paths', '.'
				, '--warn-pedantic'
				, '--warn-uninitialized'
				,'--o',tempfile()
			)
			, wd = dirname(path)
			, error_on_status = F
			, stderr_to_stdout = T
		)$stdout
		if(stdout==''){
			cat('\n')
			cat(crayon::blue('No syntax errors detected! :D\n\n'))
		}else{
			stdout = stringr::str_remove_all(stdout,stringr::fixed('./'))
			stdout = stringr::str_replace_all(stdout, 'Info: ', '\nInfo:\n')
			cat(crayon::blue(stdout))
			cat('\n\n')
		}
		invisible(NULL)
	}
	utils::assignInNamespace("rstudio_stanc", cmdstan_pedantic_check, ns="rstan", envir=as.environment("package:rstan"))
	return(invisible(NULL))
}
