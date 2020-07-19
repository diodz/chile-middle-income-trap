library(readxl)
library(dplyr)

#By Diego A. Diaz (https://github.com/diodz/chile-middle-income-trap)
#Please see published version of the article for correct citation. If not
#available, cite as:
#Couyoumdjian, JP., Larroulet, C., Diaz, D.A. (2020) Another case of the
#middle-income trap: Chile, 1900-1939. Revista de Historia Economica.

#The purpose of this section is to produce two files from the Maddison 2018
#database to perform synthetic controls in MATLAB

prepare_data <- function(){
    #Creates dataframe to be saved as csv for MATLAB
    mpd2018 <- read_excel("../data/mpd2018.xlsx", sheet = "cgdppc")

    #We delete the row of country codes
    mpd2018 <- mpd2018[-c(1),]
    #Renaming first column
    mpd2018 <- rename(mpd2018, c("year"="cgdppc"))
    chile_and_controls <- c('year', 'Chile', 'Australia', 'Canada', 'Switzerland',
                            'Denmark', 'Finland', 'Norway', 'New Zealand', 'Sweden',
                            'Portugal')
    #Keep the years that we need for synthetic controls, 1900 to 1960
    df <- mpd2018[chile_and_controls][mpd2018$year >= 1900 & mpd2018$year <= 1960,]
    df <- mutate_all(df, function(x) as.numeric(as.character(x)))
}


#We need separate files for the dataframe and countries names
df <- prepare_data()
write.csv(df, "../data/gdppc.csv", row.names=FALSE)
write.csv(colnames(df)[2:11], "../data/countries.csv", row.names=FALSE)
