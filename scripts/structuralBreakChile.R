rm(list = ls())

install.packages('strucchange')
library(strucchange)
## Nile data with one breakpoint: the annual flows drop in 1898
## because the first Ashwan dam was built
data("Nile")
plot(Nile)
## F statistics indicate one breakpoint
fs.nile <- Fstats(Nile ~ 1)
plot(fs.nile)
fs.nile$Fstats[14]

breakpoints(fs.nile)
lines(breakpoints(fs.nile))
## or
bp.nile <- breakpoints(Nile ~ 1)
summary(bp.nile)
aux <- 1:50
aux
plot(aux)
fs.aux <- Fstats(aux ~ 1)
plot(fs.aux)
summary(breakpoints(fs.aux))

summary(lm(Nile[1:28] ~ 1))
summary(lm(Nile[29:100] ~ 1))
summary(lm(Nile[1:100] ~ 1))
Nile[1:28]

anova((lm(Nile[1:50] ~ 1)))
anova(lm(Nile[51:100] ~ 1))
anova(lm(Nile[1:100] ~ 1))


169.2
124.8
135

(2835157-(593178+1819869)/1)/((593178+1819869)/(98-2))

fm0.nile <- lm(Nile ~ 1)
coef(fm0.nile)
nile.fac <- breakfactor(bp1)
fm1.nile <- lm(Nile ~ nile.fac - 1)
coef(fm1.nile)

install.packages("readxl")
library(readxl)
GapChile <- read_excel("C:/Users/diego/OneDrive - uc.cl/WORK/Data/Mit Trap/GapChile_lifeExpec.xlsx")
GapChile_lifeExpec <- read_excel("C:/Users/Diego Diaz/OneDrive - uc.cl/WORK/Data/Mit Trap/GapChile_lifeExpec.xlsx",
sheet = "Hoja2")


#chile data
n <- length(GapChile_lifeExpec$relative)
chileRel <- GapChile_lifeExpec$relative[1:n]
relative_life_exp <- ts(chileRel, start=c(1800), end=c(2016), frequency=1) 
plot(relative_life_exp)

fs.chile <- Fstats(relative_life_exp ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

data("Oil")

## or
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)

data("durab")
plot(durab)

#we do the same but with chile/USA
names(GapChile)
chileRelativeGDP <- GapChile$ChileMadisson[11:201]/GapChile$UsaMadisson[11:201]
chilets <- ts(chileRelativeGDP, start=c(1820), end=c(2010), frequency=1) 
plot(chilets)

fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

#we do the same but with chile/G7
chileRelativeGDP <- GapChile$ChileMadisson[11:201]/GapChile$UsaMadisson[11:201]
chilets <- ts(chileRelativeGDP, start=c(1820), end=c(2010), frequency=1) 
plot(chilets)


fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

#we do the same but with chile/Woffshoots
chileRelativeGDP <- GapChile$ChileMadisson[61:201]/GapChile$Waverage[61:201]
chilets <- ts(chileRelativeGDP, start=c(1870), end=c(2010), frequency=1) 
plot(chilets)

fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

## or
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)

#we do the same but with the gap
chileGAP <- GapChile$GAPmadisson[11:201]/GapChile$ChileMadisson[11:201]
chilets <- ts(chileGAP, start=c(1820), end=c(2010), frequency=1) 
plot(chilets)

fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

## or
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)


#growth percentage
chileGrowth <- (GapChile$ChileMadisson[12:201]-
                  GapChile$ChileMadisson[11:200])/GapChile$ChileMadisson[12:201]
chilets <- ts(chileGrowth, start=c(1820), end=c(2010), frequency=1) 
plot(chilets)

fs.chile <- Fstats(chilets ~ 1)
plot(fs.chile)
breakpoints(fs.chile)
lines(breakpoints(fs.chile))

## or
bp.chile <- breakpoints(chilets ~ 1)
summary(bp.chile)
