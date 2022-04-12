cat('get_contrast_matrix_rows_as_list() now available as a function\n')
get_contrast_matrix_rows_as_list = function(
		data
		, formula
		, contrast_kind = NULL
		, safe_names_prefix = '__'
){
	formula_vars = all.vars(formula)
	mm = get_contrast_matrix(data, formula, contrast_kind )
	true_names = dimnames(mm)[[2]]
	safe_names = paste0(safe_names_prefix,dimnames(mm)[[2]])
	mm_list = lapply(
		X = seq_len(nrow(mm))
		, FUN = function(i){
			out = as_tibble(array(
				mm[i,]
				, dim = c(1,ncol(mm))
				, dimnames = list(NULL,safe_names)
			))
			names(out) = safe_names
			attr(out,'true_names') = true_names
			attr(out,'formula_vars') = formula_vars
			return(out)
		}
	)
	return(mm_list)
}
