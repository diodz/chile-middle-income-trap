library(readxl)
library(plyr)
library(dplyr)

mpd2018 <- read_excel("data/mpd2018.xlsx",
sheet = "cgdppc")

#We delete the row of country codes
mpd2018 <- mpd2018[-c(1),]

#Renaming first column
mpd2018 <- rename(mpd2018, c("cgdppc"="year"))

chile_and_controls <- c('year', 'Chile', 'Australia', 'Canada', 'Switzerland',
                        'Denmark', 'Finland', 'Norway', 'New Zealand', 'Sweden',
                        'Portugal')


df <- mpd2018[chile_and_controls][mpd2018$year >= 1900 & mpd2018$year <= 1960,]

df <- mutate_all(df, function(x) as.numeric(as.character(x)))

write.csv(df, "../data/gdppc.csv", row.names=FALSE)

write.csv(colnames(df)[2:11], "../data/countries.csv", row.names=FALSE)

#europe12 <- c()

#df[c("A","B","E")]