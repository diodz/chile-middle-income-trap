library(readxl)
library(dplyr)
library(strucchange)
library(ggplot2)
source('relative_income_figure_1.R')

#Structural break in the time series of relative income between Chile
#and the United States. We begin the analysis in the year 1820.
#First step is loading the data.
usa_rel <- get_relative_usa(1820)

#We create helper functions to easily plot our results
plot_time_series <- function(df, column_name, start_year, figure_filename){
    file_path <- paste('../figures/', figure_filename, '.png', sep='')
    time_series <- ts(df[df$year >= start_year,][[column_name]],
                      start=c(start_year), end=c(2016), frequency=1)
    png(file_path, width = 450, height = 350)
    plot(time_series, lwd = 2, xlab = 'Year', ylab = 'Relative GDPPC')
    dev.off()
}

plot_Fstat_test <- function(df, column_name, start_year, figure_filename){
    file_path <- paste('../figures/', figure_filename, '.png', sep='')
    time_series <- ts(df[df$year >= start_year,][[column_name]],
                      start=c(start_year), end=c(2016), frequency=1)
    fs.chile <- Fstats(time_series ~ 1)
    png(file_path, width = 450, height = 350)
    plot(fs.chile, xlab = 'Year', lwd = 3)
    breakpoints(fs.chile)
    lines(breakpoints(fs.chile))
    dev.off()
}

#We also create a helper function to do a Bai Perron structural break test
#for robustness
bai_perron <- function(df, column_name, start_year){
    time_series <- ts(df[df$year >= start_year,][[column_name]],
                      start=c(start_year), end=c(2016), frequency=1)
    bp.test <- breakpoints(time_series ~ 1)
    summary(bp.test)
}

#Figure 2_1
plot_time_series(usa_rel, 'chile_relative', 1820, 'Figure 2_1')

#Figure 2_2
plot_Fstat_test(usa_rel, 'chile_relative', 1820, 'Figure 2_2')

#Bai Perron Table
bai_perron(usa_rel, 'chile_relative', 1820)

#--------------------------------------------------------------

#Next we do the same with the relative income series of Chile and the
#average of Europe 12 and Western Offshoots countries.

#Figure 2_3
plot_time_series(grouped, 'e12_wo_relative', 1880, 'Figure 2_3')

#Figure 2_4
plot_Fstat_test(grouped, 'e12_wo_relative', 1880, 'Figure 2_4')

#--------------------------------------------------------------

#Next we obtain the average of the Nordic3 countries from Maddison to
#perform structural break tests with the relative income time series of
#Chile over Nordic3

get_Nordic3_relative <- function(){
    #Read income data
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
    #rm(cgdppc, population)

    #Get relative income using function from relative_income.R
    nordic3_grouped <- relative_income(nordic3_income, nordic3_pop)
    #rm(nordic3_income, nordic3_pop, nordic3)
    return(nordic3_grouped)
}

#Next we load the Nordic 3 data.
nordic3 <- get_Nordic3_relative()

#Now we can perform Chow and Bai Perron's tests with the time series of
#relative income between Chile and the average of the Nordic3 countries.
#We start in the year 1880 for consistency.

#Figure 2_5
plot_time_series(nordic3, 'chile_relative', 1880, 'Figure 2_5')

#Figure 2_6
plot_Fstat_test(nordic3, 'chile_relative', 1880, 'Figure 2_6')

#Bai Perron Table
bai_perron(nordic3, 'chile_relative', 1880)
