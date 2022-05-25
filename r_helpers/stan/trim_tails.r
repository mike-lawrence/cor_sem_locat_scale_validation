cat('trim_tails() now available as a function\n')
trim_tails = function(value,p=.01){
	value[(value>quantile(value,p))&(value<quantile(value,1-p))]
}
