library(readxl)
library(dplyr)

#The purpose of this section is to produce two files from the Maddison 2018
#database to perform synthetic controls in MATLAB

cgdppc <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")

#We delete the row of country codes
cgdppc <- cgdppc[-c(1),]

#Renaming first column
cgdppc <- rename(cgdppc, c("year"="cgdppc"))

#We extract the western offshoots countries for comparison
western_offshoots <- c('Australia', 'New Zealand', 'Canada', 'United States')
cgdppc$year <- as.numeric(cgdppc$year)
wo_income <- cgdppc[c('year', 'Chile', western_offshoots)][cgdppc$year
                                                            >= 1870,]
wo_income <- mutate_all(wo_income, function(x) as.numeric(as.character(x)))

#Read population data
population <- read_excel("../data/mpd2018.xlsx", sheet = "pop")
population <- population[-c(1),]
population <- rename(population, c("year"="pop"))
population$year <- as.numeric(population$year)
wo_pop <- population[c('year', 'Chile', western_offshoots)][population$year
                                                         >= 1870,]
wo_pop <- mutate_all(wo_pop, function(x) as.numeric(as.character(x)))

#We extract the Europe 12s countries for comparison with Chile
europe12 <- c('Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany',
              'Italy', 'Netherlands', 'Norway', 'Sweden',
              'Switzerland', 'United Kingdom')
e12_income <- cgdppc[c('year', 'Chile', europe12)]
e12_income <- mutate_all(e12_income, function(x) as.numeric(as.character(x)))
e12_income <- e12_income[e12_income$year >= 1870,]

e12_pop <- population[c('year', 'Chile', europe12)]
e12_pop <- mutate_all(e12_pop, function(x) as.numeric(as.character(x)))
e12_pop <- e12_pop[e12_pop$year >= 1870,]

#We remove loaded data from workspace
rm(cgdppc, population)

#We write a function to get the relative income of a group of countries to
#compare with Chile
relative_income <- function(df_income, df_pop){
    df_relative <- df_income
    countries <- colnames(df_income)[3:length(colnames(df_income))]
    df_relative$total_gdp <- 0
    df_relative$total_pop <- 0
    for (country in countries){
        df_relative[[country]] <- df_income[[country]] * df_pop[[country]]
        df_relative$total_gdp <- df_relative$total_gdp + df_relative[[country]]
        df_relative$total_pop <- df_relative$total_pop + df_pop[[country]]
    }
    df_relative$group_gdppc <- df_relative$total_gdp / df_relative$total_pop
    df_relative <- df_relative[c('year', 'Chile', 'group_gdppc')]
    df_relative$chile_relative <- df_relative$Chile / df_relative$group_gdppc
    return(df_relative)
}

#Getting average gdppc for Europe 12
e12_grouped <- relative_income(e12_income, e12_pop)
wo_grouped <- relative_income(wo_income, wo_pop)

#We remove previous data from workspace
rm(e12_income, e12_pop, wo_income, wo_pop)

#Merging dataframes to plot in the same figure
e12_grouped <- rename(e12_grouped, c("e12_relative"="chile_relative"))
wo_grouped <- rename(wo_grouped, c("wo_relative"="chile_relative"))
e12_grouped <- rename(e12_grouped, c("e12"="group_gdppc"))
wo_grouped <- rename(wo_grouped, c("wo"="group_gdppc"))
grouped <- merge(e12_grouped, wo_grouped[-c(2)], by='year')
