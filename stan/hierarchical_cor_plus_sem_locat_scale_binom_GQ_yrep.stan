functions{
	// flatten_lower_tri: function that returns the lower-tri of a matrix, flattened to a vector
	vector flatten_lower_tri(matrix mat) {
		int n_cols = cols(mat) ;
		int n_uniq = (n_cols * (n_cols - 1)) %/% 2;
		vector[n_uniq] out ;
		int i = 1;
		for(c in 1:(n_cols-1)){
			for(r in (c+1):n_cols){
				out[i] = mat[r,c];
				i += 1;
			}
		}
		return(out) ;
	}


	matrix sem_var1_to_var2(
		matrix std_normal_var1
		, vector cors
		, matrix std_normal_var2_unique
	){
		int nrows = rows(std_normal_var2_unique) ;
		int ncols = cols(std_normal_var2_unique) ;
		vector[nrows] vars_common = cors.^2 ;
		vector[nrows] vars_unique = 1-vars_common ;
		matrix[nrows,ncols] std_normal_var2 = (
			(
				(
					std_normal_var1[1:nrows,]
					.* rep_matrix(
						cors
						, ncols
					)
				)
				+ (
					std_normal_var2_unique
					.* rep_matrix(
						sqrt(vars_unique)
						, ncols
					)
				)
			)
			// divide by the square-root of the sum of the squared weights to yield unit-scale variates (since the component variates have unit-scale too)
			./ rep_matrix(
				sqrt( vars_common + vars_unique )
				, ncols
			)
		) ;
		return(std_normal_var2) ;
	}

	matrix shift_and_scale_cols(
		matrix std_normal_vals
		, vector shift
		, vector scale
	){
		int nrows = rows(std_normal_vals) ;
		int ncols = cols(std_normal_vals) ;
		matrix[nrows,ncols] shifted_and_scaled_vals = (
			rep_matrix(shift,ncols)
			+ (
				std_normal_vals
				.* rep_matrix(scale,ncols)
			)
		) ;
		return(shifted_and_scaled_vals) ;
	}


}

data{

	// nI: number of individuals
	int<lower=2> nI ;

	// nXc: number of condition-level predictors
	int<lower=2> nXc ;

	// rXc: number of rows in the condition-level predictor matrix
	int<lower=nXc> rXc ;

	// Xc: condition-level predictor matrix
	matrix[rXc,nXc] Xc ;

	// iXc: which individual is associated with each row in Xc
	array[rXc] int<lower=1,upper=nI> iXc ;

	// nY: num entries in the observation vector
	int nY ;

	// Y_gauss: observations modelled with location-scale Gaussian model
	vector[nY] Y_gauss ;

	// Y_binom: observations modelled with binomial model
	array[nY] int<lower=0,upper=1> Y_binom ;

	// yXc: which row in Xc is associated with each observation in Y
	array[nY] int<lower=1,upper=rXc> yXc ;

}

transformed data{

	matrix[nXc,rXc] Xct = transpose(Xc) ;

}

parameters{

	cholesky_factor_corr[nXc] locat_cholfaccorr ;
	matrix[nXc,nI] locat_icoef_std_normals ;
	vector[nXc] locat_coef_mean ;
	vector<lower=0>[nXc] locat_coef_sd ;

	vector<lower=-1,upper=1>[nXc] locat_scale_cors ;
	matrix[nXc,nI] scale_icoef_unique_std_normals ;
	vector[nXc] scale_coef_mean ;
	vector<lower=0>[nXc] scale_coef_sd ;

	vector<lower=-1,upper=1>[nXc] locat_binom_cors ;
	matrix[nXc,nI] binom_icoef_unique_std_normals ;
	vector[nXc] binom_coef_mean ;
	vector<lower=0>[nXc] binom_coef_sd ;


}
generated quantities{
	// Y_gauss_rep: posterior-predictive observations modelled with location-scale Gaussian model
	vector[nY] Y_gauss_rep ;

	// Y_binom_rep: posterior-predictive observations modelled with binomial model
	array[nY] int<lower=0,upper=1> Y_binom_rep ;

	{
		// corStdNorms from cors & std-normal ----
		matrix[nXc,nI] locat_icoef_corStdNorms = (
			locat_cholfaccorr
			* locat_icoef_std_normals
		) ;

		// prep to loop over time
		matrix[nXc,nI] locat_icoef ;
		matrix[nXc,nI] scale_icoef ;
		matrix[nXc,nI] binom_icoef ;
		row_vector[rXc] locat_icond ;
		row_vector[rXc] scale_icond ;
		row_vector[rXc] binom_icond ;

		// locat just needs shift & scale
		locat_icoef = shift_and_scale_cols(
			locat_icoef_corStdNorms // std_normal_vals
			, locat_coef_mean // shift
			, locat_coef_sd // scale
		) ;
		// scale needs SEM from locat, then shift & scale
		scale_icoef = shift_and_scale_cols(
			sem_var1_to_var2(
				locat_icoef_corStdNorms // std_normal_var1
				, locat_scale_cors // cors
				, scale_icoef_unique_std_normals // std_normal_var2_unique
			) // std_normal_vals
			, scale_coef_mean // shift
			, scale_coef_sd // scale
		) ;
		// binom needs SEM from locat, then shift & scale
		binom_icoef = shift_and_scale_cols(
			sem_var1_to_var2(
				locat_icoef_corStdNorms // std_normal_var1
				, locat_binom_cors // cors
				, binom_icoef_unique_std_normals // std_normal_var2_unique
			) // std_normal_vals
			, binom_coef_mean // shift
			, binom_coef_sd // binom
		) ;
		// dot products to go from coef to cond
		locat_icond = columns_dot_product(	locat_icoef[,iXc] , Xct ) ;
		scale_icond = sqrt(exp(columns_dot_product( scale_icoef[,iXc] , Xct ))) ;
		binom_icond = columns_dot_product(	binom_icoef[,iXc] , Xct ) ;

		for(i_nY in 1:nY){
			Y_gauss_rep[i_nY] = normal_rng(
				locat_icond[yXc[i_nY]]
				, scale_icond[yXc[i_nY]]
			) ;
			Y_binom_rep[i_nY] = bernoulli_logit_rng(
				binom_icond[yXc[i_nY]]
			) ;
		}

	}

}
