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

	array [] matrix sem_var0_to_var1_var2(
		matrix std_normal_var0
		, vector cors
		, array [] matrix std_normal_var1_var2_unique
	){
		int nrows = rows(std_normal_var0) ;
		int ncols = cols(std_normal_var0) ;
		vector[nrows] vars_common = pow(cors,2) ;
		vector[nrows] vars_unique = 1-vars_common ;
		vector[nrows] sqrt_vars_unique = sqrt(vars_unique) ;
		vector[nrows] sqrt_vars_sum = sqrt(vars_unique+vars_common) ;
		matrix[nrows,ncols] std_normal_var0_times_cors = std_normal_var0 .* rep_matrix(cors,ncols) ;
		array[2] matrix[nrows,nrows] sem_stdnorms_out ;
		for(i in 1:2){
			sem_stdnorms_out[i] = (
				(
					std_normal_var0_times_cors
					+ ( std_normal_var1_var2_unique[i] .* rep_matrix(sqrt_vars_unique,ncols) )
				)
				// divide by the square-root of the sum of the squared weights to yield unit-scale variates (since the component variates have unit-scale too)
				./ rep_matrix( sqrt_vars_sum , ncols )
			) ;
		}
		return(sem_stdnorms_out);
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
	real dummy ;
}

generated quantities{

	matrix[nXc,nXc] locat_cholfaccorr ;
	matrix[nXc,nI] locat_icoef_std_normals ;
	vector[nXc] locat_coef_mean ;
	vector<lower=0>[nXc] locat_coef_sd ;

	vector<lower=-1,upper=1>[nXc] locat_scale_cors ;
	matrix[nXc,nI] scale_icoef_unique_std_normals ;
	vector[nXc] scale_coef_mean ;
	vector<lower=0>[nXc] scale_coef_sd ;

	vector<lower=-1,upper=1>[nXc] locat_binom_cors ;
	real<lower=0,upper=1> locat_binom_cors_intercept ;
	vector<lower=-1,upper=1>[nXc-1] locat_binom_cors_others ;
	matrix[nXc,nI] binom_icoef_unique_std_normals ;
	vector[nXc] binom_coef_mean ;
	vector<lower=0>[nXc] binom_coef_sd ;

	locat_cholfaccorr = lkj_corr_cholesky_rng(nXc,1.0) ;

	for(x in 1:nXc){
		locat_scale_cors[x] = uniform_rng(-1,1) ;
		locat_binom_cors[x] = uniform_rng(-1,1) ;
		locat_coef_mean[x] = std_normal_rng() ;
		scale_coef_mean[x] = std_normal_rng() ;
		binom_coef_mean[x] = std_normal_rng() ;
		locat_coef_sd[x] = weibull_rng(2,1) ;
		scale_coef_sd[x] = weibull_rng(2,1) ;
		binom_coef_sd[x] = weibull_rng(2,1) ;
		for(i in 1:nI){
			locat_icoef_std_normals[x,i] = std_normal_rng() ;
			scale_icoef_unique_std_normals[x,i] = std_normal_rng() ;
			binom_icoef_unique_std_normals[x,i] = std_normal_rng() ;
		}
	}
	locat_binom_cors[1] = uniform_rng(0,1) ;
	locat_binom_cors_intercept = locat_binom_cors[1] ;
	locat_binom_cors_others = locat_binom_cors[2:nXc] ;

	vector[(nXc*(nXc-1))%/%2] locat_cors = flatten_lower_tri(multiply_lower_tri_self_transpose(locat_cholfaccorr)) ;

}
