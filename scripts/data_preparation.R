library(readxl)
library(dplyr)
library(ggplot2)

#The purpose of this section is to compare Chile's relative income with the
#the average of the Western Europe countries and Western Offshoots and also
#with the USA. Produces Figure 1.

relative_income <- function(df_income, df_pop){
    #This function gets the relative income of a group of countries
    #in a given dataframe to compare with Chile
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

load_maddison_gdppc <- function(){
    #Loads Maddison's 2018 cgdppc data for all countries and does some basic
    #data cleaning.
    df <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")
    df <- df[-c(1),]
    df <- rename(df, c("year"="cgdppc"))
    return(df)
}

load_maddison_pop <- function(){
    #Loads Maddison's 2018 population data for all countries and does some
    #basic data cleaning.
    pop <- read_excel("../data/mpd2018.xlsx", sheet = "pop")
    pop <- pop[-c(1),]
    pop <- rename(pop, c("year"="pop"))
    pop$year <- as.numeric(pop$year)
    return(pop)
}

get_relative_usa <- function(year){
    #Gets the relative gdp per capita between Chile and the United States
    cgdppc <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")
    cgdppc <- cgdppc[-c(1),]
    cgdppc <- rename(cgdppc, c("year"="cgdppc"))
    cgdppc$year <- as.numeric(cgdppc$year)
    small <- cgdppc[c('year', 'Chile', 'United States')][cgdppc$year
                                                         >= year,]
    small <- mutate_all(small, function(x) as.numeric(as.character(x)))
    small$chile_relative <- small[['Chile']] / small[['United States']]
    return(small[c('year', 'chile_relative')])
}

get_western_offshoots <- function(){
    #This function obtains the relative income of chile and the western
    #offshoots countries
    cgdppc <- load_maddison_gdppc()
    population <- load_maddison_pop()

    #We extract the western offshoots countries
    western_offshoots <- c('Australia', 'New Zealand', 'Canada',
                           'United States')
    cgdppc$year <- as.numeric(cgdppc$year)
    wo_income <- cgdppc[c('year', 'Chile', western_offshoots)][cgdppc$year
                                                               >= 1870,]
    wo_income <- mutate_all(wo_income, function(x) as.numeric(as.character(x)))

    wo_pop <- population[c('year', 'Chile', western_offshoots)][population$year
                                                                >= 1870,]
    wo_pop <- mutate_all(wo_pop, function(x) as.numeric(as.character(x)))
    return(relative_income(wo_income, wo_pop))
}

get_e12 <- function(){
    #This function obtains the relative income of chile and the western
    #offshoots countries
    cgdppc <- load_maddison_gdppc()
    population <- load_maddison_pop()

    #We extract the western offshoots countries
    europe12 <- c('Austria', 'Belgium', 'Denmark', 'Finland', 'France',
                  'Germany', 'Italy', 'Netherlands', 'Norway', 'Sweden',
                  'Switzerland', 'United Kingdom')
    e12_income <- cgdppc[c('year', 'Chile', europe12)]
    e12_income <- mutate_all(e12_income, function(x) as.numeric(as.character(x)))
    e12_income <- e12_income[e12_income$year >= 1870,]

    e12_pop <- population[c('year', 'Chile', europe12)]
    e12_pop <- mutate_all(e12_pop, function(x) as.numeric(as.character(x)))
    e12_pop <- e12_pop[e12_pop$year >= 1870,]
    return(relative_income(e12_income, e12_pop))
}

average_wo_e12 <- function(){
    #Gets relative income between Chile and the simple average of Europe 12 and
    #Western Offshoots.
    e12 <- get_e12()
    wo <- get_western_offshoots()
    e12 <- rename(e12, c("e12_relative"="chile_relative"))
    wo <- rename(wo, c("wo_relative"="chile_relative"))
    e12 <- rename(e12, c("e12"="group_gdppc"))
    wo <- rename(wo, c("wo"="group_gdppc"))
    df <- merge(e12, wo[-c(2)], by='year')
    df$e12_wo_cgdppc <- (df$e12 + df$wo)/2
    df$chile_relative <- df$Chile / df$e12_wo_cgdppc
    return(df[c('year', 'Chile', 'e12_wo_cgdppc', 'chile_relative')])
}

ggplot_relative_income <- function(){
    #Putting aggregated relative series in the same dataframe for easier plotting
    usa_rel <- get_relative_usa(1870)
    grouped <- average_wo_e12()
    grouped$chile_usa_relative <- usa_rel$chile_relative

    #Finally we can plot the relative income of Chile versus the average of Europe
    #12 and Western Offshoots and the USA
    ggplot(grouped, aes(year)) + geom_line(aes(y = chile_relative, colour =
                                                   'chile_relative'), size=1.2)+
        geom_line(aes(y = chile_usa_relative, colour = 'chile_usa_relative'),
                    size=1.2) +
        scale_x_continuous(breaks = round(seq(1870, 2010, by = 20),1)) +
        scale_y_continuous(breaks = round(seq(0, 0.9, by = 0.1),1),
                           limits = c(0.1, 0.7)) +
        scale_color_discrete(name=element_blank(), labels=
                                 c('Chile/Avg WO - E12','Chile/USA')) +
        theme(legend.position = c(0.72, 0.68), text=element_text(size=20),
             panel.background = element_rect(fill = 'white', colour = 'black'))+
        geom_hline(yintercept=0.6, linetype="dashed", color = "black") +
        geom_hline(yintercept=0.15, linetype="dashed", color = "black") +
        xlab('Year') + ylab('Relative income')

    #We save the plot in the figures folder
    ggsave(width = 8, height = 5, dpi = 300, filename =
               '../figures/Figure 1.png')
}

