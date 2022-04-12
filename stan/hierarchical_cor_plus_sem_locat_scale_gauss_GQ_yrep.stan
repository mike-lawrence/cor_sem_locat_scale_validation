/*
aria: compile = 1
aria: make_local += STAN_NO_RANGE_CHECKS=true
aria: make_local += STAN_CPP_OPTIMS=true
// aria: make_local += CXXFLAGS+=-mtune=native
// aria: make_local += CXXFLAGS+=-march=native
// aria: make_local += STANCFLAGS+=--O1
// aria: make_local += CXXFLAGS+=-O3
// aria: make_local += CXXFLAGS+=-g0
// aria: make_local += STAN_NO_RANGE_CHECKS=true
// aria: make_local += STAN_CPP_OPTIMS=true
// aria: make_local += STANCFLAGS+=--O1
// aria: make_local += STANCFLAGS+=--Oexperimental
// aria: make_local += PRECOMPILED_HEADERS=true
*/

////
// Glossary
////

// n.b. the following employs a mix of snake_case and camelCase that is sure to
//   vex some, but represents the author's best attempt to balance to the competing
//   aims of clarity & brevity.

// Y: observed outcome
// nY: number of observed outcomes
// X: predictor/contrast matrix
// nX: number of predictors (columns in the contrast matrix)
// rX: number of rows in the contrast matrix X
// (i)ndividual: a unit of observation within which correlated measurements may take place
// (c)ondition: a labelled set of observations within an individual that share some feature/predictor or conjunction of features/predictors
// Xc: condition-level contrast matrix
// nXc: number of predictors in the condition-level contrast matrix
// rXc: number of rows in the condition-level contrast matrix
// yXc: for each observation in y, an index indicating the associated row in Xc corresponding to that observation's individual/condition combo
// Z: matrix of coefficient row-vectors to be dot-product'd with a contrast matrix
// indiv_locat_coef: matrix of coefficient row-vectors associated with each individual

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

	matrix var1_var2_corsem(matrix std_normal_var1, vector cors, matrix std_normal_var2_unique){
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

	matrix shift_and_scale_cols(matrix std_normal_vals, vector shift, vector scale){
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
	int<lower=0,upper=1> prior_informed ;
	int<lower=0,upper=1> likelihood_informed ;

	// nI: number of individuals
	int<lower=2> nI ;

	// nXc: number of condition-level predictors
	int<lower=2> nXc ;

	// LA variables ----

	// rXc: number of rows in the condition-level predictor matrix
	int<lower=nXc> rXc ;

	// Xc: condition-level predictor matrix
	matrix[rXc,nXc] Xc ;

	// iXc: which individual is associated with each row in Xc
	array[rXc] int<lower=1,upper=nI> iXc ;

	// nY: num entries in the LA observation vector
	int nY ;

	// Y: LA observations
	vector[nY] Y ;

	// yXc: which row in Xc is associated with each observation in Y
	array[nY] int<lower=1,upper=rXc> yXc ;

}

transformed data{

	matrix[nXc,rXc] Xct = transpose(Xc) ;

}

parameters{

	// real<lower=0> locat_intercept_mean ;
	real locat_intercept_mean ;
	real<lower=0> locat_intercept_sd ;
	vector[nXc-1] locat_coef_mean ;
	vector<lower=0>[nXc-1] locat_coef_sd ;


	matrix[nXc,nI] locat_icoef_indiv_helper ;
	cholesky_factor_corr[nXc] locat_cholfaccorr ;

	// real<lower=0> scale_intercept_mean ;
	real scale_intercept_mean ;
	real<lower=0> scale_intercept_sd ;
	vector[nXc-1] scale_coef_mean ;
	vector<lower=0>[nXc-1] scale_coef_sd ;

	vector<lower=-1,upper=1>[nXc] locat_scale_cors ;
	matrix[nXc,nI] scale_icoef_indiv_unique ;

}
generated quantities{
	// Y_rep: posterior-predictive LA observations
	vector[nY] Y_rep ;
	{ // local environment to avoid saving intermediate quantities
		// corStdNorms from cors & std-normal helpers ----
		matrix[nXc,nI] locat_icoef_indiv_corStdNorms = (
			locat_cholfaccorr
			* locat_icoef_indiv_helper
		) ;

		// Shifting & scaling ----
		matrix[nXc,nI] locat_icoef_indiv = shift_and_scale_cols(
			locat_icoef_indiv_corStdNorms // std_normal_vals
			, append_row( locat_intercept_mean , locat_coef_mean ) // shift
			, append_row( locat_intercept_sd , locat_coef_sd ) // scale
		) ;


		// dot products ----
		row_vector[rXc] locat_cond = columns_dot_product(
			locat_icoef_indiv[,iXc]
			, Xct
		) ;

		// SEMs ----
		matrix[nXc,nI] scale_icoef_indiv_corStdNorms = var1_var2_corsem(
			locat_icoef_indiv_corStdNorms // std_normal_var1
			, locat_scale_cors // cors
			, scale_icoef_indiv_unique // std_normal_var2_unique
		) ;

		// Shifting & scaling ----
		matrix[nXc,nI] scale_icoef_indiv = shift_and_scale_cols(
			scale_icoef_indiv_corStdNorms // std_normal_vals
			, append_row( scale_intercept_mean , scale_coef_mean ) // shift
			, append_row( scale_intercept_sd , scale_coef_sd ) // scale
		) ;

		// dot products ----
		row_vector[rXc] scale_cond = sqrt(exp(columns_dot_product(
			scale_icoef_indiv[,iXc]
			, Xct
		))) ;

		for(i_nY in 1:nY){
			Y_rep[i_nY] = normal_rng(
				locat_cond[yXc[i_nY]]
				, scale_cond[yXc[i_nY]]
			) ;
		}

	}
}
