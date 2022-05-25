cat('start_ramdisk() now available as a function\n')
start_ramdisk = function(gb,path='/tmp/ramdisk'){
	if(is.null(path)){
		path = getOption('ramdisk_path')
		if(is.null(path)){
			path = '/tmp/ramdisk'
		}
	}
	orig_warn = options(warn=-1)
	pwfile = tempfile()
	write(rstudioapi::askForPassword("sudo password"),pwfile)
	try(
		out <- system2(
			command = 'sudo'
			, args = paste0(
				"-kS sh ./r_helpers/shell_scripts/start_ramdisk.sh "
				, dQuote(path,q=F)
				, " "
				, gb
				, "G"
			)
			, stdin = pwfile
			, stdout = TRUE
		)
		, silent = TRUE
	)
	cat(' (captured via rstudioapi::askForPassword)\n')
	fs::file_delete(pwfile)
	options(orig_warn)
	expected_out = paste0("ramdisk of size ",gb,"G successfully started at ",dQuote(path,q=F))
	if(out==expected_out){
		options('ramdisk_path'=path)
		cat(crayon::cyan(out))
	}else{
		cat(crayon::red(out))
	}
	return(invisible(NULL))
}


# if(!fs::dir_exists('/tmp/ramdisk')){
# 	system("sudo -kS ~/.ramdisk/start 30G", input = rstudioapi::askForPassword("sudo password"))
# }
#system("sudo -kS ~/.ramdisk/stop", input = rstudioapi::askForPassword("sudo password"))
