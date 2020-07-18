library(readxl)
library(dplyr)

#The purpose of this section is to produce two files from the Maddison 2018
#database to perform synthetic controls in MATLAB

mpd2018 <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")

#We delete the row of country codes
mpd2018 <- mpd2018[-c(1),]

#Renaming first column
mpd2018 <- rename(mpd2018, c("year"="cgdppc"))

chile_and_western_offshoots <- c('year', 'Chile', 'Australia', 'New Zealand',
                                 'Canada', 'United States')

#Keep the years that we need for synthetic controls, 1900 to 1960
mpd2018$year <- as.numeric(mpd2018$year)
cgdppc <- mpd2018[chile_and_western_offshoots][mpd2018$year >= 1810,]
cgdppc <- mutate_all(df, function(x) as.numeric(as.character(x)))

#Read population data
mpd2018 <- read_excel("../data/mpd2018.xlsx", sheet = "pop")
mpd2018 <- mpd2018[-c(1),]
mpd2018$year <- as.numeric(mpd2018$year)
mpd2018 <- rename(mpd2018, c("year"="pop"))
pop <- mpd2018[chile_and_western_offshoots][mpd2018$year >= 1810,]


