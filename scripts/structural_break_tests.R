library(readxl)
library(dplyr)
library(strucchange)
source('relative_income_figure_1.R')

#Structural break in the time series of relative income between Chile
#and the United States. We begin the analysis in the year 1820.
usa_rel <- relative_usa(1820)
chilets <- ts(usa_rel$chile_relative, start=c(1820), end=c(2016), frequency=1)
plot(chilets)
fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

#Bai Perron Table
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)

#Next we obtain the average of the Nordic3 countries from Maddison to
#perform structural break tests with the relative income time series of
#Chile over Nordic3
cgdppc <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")
cgdppc <- cgdppc[-c(1),]
cgdppc <- rename(cgdppc, c("year"="cgdppc"))
cgdppc$year <- as.numeric(cgdppc$year)
nordic3 <- c('Sweden', 'Norway', 'Finland')

nordic3_income <- cgdppc[c('year', 'Chile', nordic3)][cgdppc$year
                                                           >= 1870,]
nordic3_income <- mutate_all(nordic3_income, function(x)
    as.numeric(as.character(x)))

#Read population data
population <- read_excel("../data/mpd2018.xlsx", sheet = "pop")
population <- population[-c(1),]
population <- rename(population, c("year"="pop"))
population$year <- as.numeric(population$year)
nordic3_pop <- population[c('year', 'Chile', nordic3)][population$year
                                                            >= 1870,]
nordic3_pop <- mutate_all(nordic3_pop, function(x) as.numeric(as.character(x)))

#We remove loaded data from workspace
rm(cgdppc, population)

nordic3_grouped <- relative_income(nordic3_income, nordic3_pop)
rm(nordic3_income, nordic3_pop, nordic3)

#Structural break in the time series of relative income between Chile
#and the average of the Nordic3 countries. We start in the year 1880.
chilets <- ts(nordic3_grouped[nordic3_grouped$year>=1880,]$chile_relative,
              start=c(1880), end=c(2016), frequency=1)
plot(chilets)
fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

#Bai Perron Table
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)

#Next we do the same with the relative income series of Chile and the
#Europe 12 countries.
chilets <- ts(grouped[grouped$year>=1880,]$e12_wo_relative, start=c(1880),
              end=c(2016), frequency=1)
plot(chilets)
fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)
