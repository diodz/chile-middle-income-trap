# assuming you have a 'ts' object in R 

# 1. install package 'strucchange'
# 2. Then write down this code:

install.packages('strucchange')
library(strucchange)

# store the breakdates
bp_ts <- breakpoints(ts)

# this will give you the break dates and their confidence intervals
summary(bp_ts) 

# store the confidence intervals
ci_ts <- confint(bp_ts)

## to plot the breakpoints with confidence intervals
plot(ts)
lines(bp_ts)
lines(ci_ts)

aux <- c(1:50,51:1)
plot(aux)

length(aux)

auxts <- ts(aux, start=c(1900), end=c(2000), frequency=1) 
plot(auxts)
length(auxts)

t <- seq_along(auxts)

fs.aux <- Fstats(auxts ~ 1+t)
lm(auxts[51:100] ~ 1+t[51:100])

plot(fs.aux)
breakpoints(fs.aux)
lines(breakpoints(fs.aux))
