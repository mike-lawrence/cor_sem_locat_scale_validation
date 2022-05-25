options(warn=1) # really should be default in R
`%!in%` = Negate(`%in%`) #should be in base R!

# specify the packages used:
required_packages = c(
	'fs' # for file ops
	, 'cmdstanr' # for stan
	, 'tidyverse' # for all that is good and holy
)

# load the helper functions:
source('r_helpers/install_if_missing.r') # defines install_if_missing()

# install any required packages not already present
install_if_missing(required_packages)

# load the tidyverse
options(tidyverse.quiet = TRUE)
library(tidyverse)
future::plan('multisession')

# helpers
for(helper_file in fs::dir_ls('r_helpers',recurse=T,type = 'file',regex='\\.[rR]')){
	cat(paste0('\nSourcing "',helper_file,'"\n'))
	source(helper_file)
}
