setwd("C:/Users/diego/OneDrive - uc.cl/WORK/R/Synthethic control/ChileSynth") #notebook
setwd("C:/Users/Diego Diaz/OneDrive - uc.cl/WORK/R/Synthethic control/ChileSynth") #desktop

rm(list = ls()) #remove all variables in environment

install.packages("data.table")

library(Synth)
library(data.table)
library(readr)
library(readxl)

Allgdp_barro_polityII <- read_excel("C:/Users/diego/OneDrive - uc.cl/WORK/Premium datasets/Merged/Allgdp_barro_polityII.xlsx")
Allgdp_barro_polityII <- read_excel("C:/Users/Diego Diaz/OneDrive - uc.cl/WORK/Premium datasets/Merged/Allgdp_barro_polityII.xlsx")


class(Allgdp_barro_polityII)
AUX <- as.data.table(Allgdp_barro_polityII)
class(AUX)
rm(Allgdp_barro_polityII)

#we save our datatable object for the future
saveRDS(AUX, "diazSynth.rds")

diazSynth <- readRDS("diazSynth.rds")


year0 <- 1935
yearT <- 1935

### chunk number 3
###################################################
           dataprep.out <-
             dataprep(foo = AUX, 
                      predictors = c("polity2" , "durable" , "xconst"),
                      predictors.op = "mean" ,
                      time.predictors.prior = 1900:1938,
                        special.predictors = list(
                        list("NoSchooling" ,   seq(year0,yearT,5), "mean"),
                         list("Primary_total" ,  seq(year0,yearT,5), "mean"),
                        list("Secondary_total" ,   seq(year0,yearT,5), "mean"),
                        list("Tertiary_total" ,   seq(year0,yearT,5), "mean"),
                        list("Primary_completed" ,   seq(year0,yearT,5), "mean"),
                        list("Secondary_completed" ,   seq(year0,yearT,5), "mean"),
                        list("Tertiary_completed" ,  seq(year0,yearT,5), "mean"),
                        list("AvgYearsofTotalSchooling" ,   seq(year0,yearT,5), "mean"),
                        list("AvgYearsofPrimarySchooling" ,   seq(year0,yearT,5), "mean"),
                        list("AvgYearsofSecondarySchooling" , seq(year0,yearT,5), "mean"),
                        list("AvgYearsofTertiarySchooling" ,   seq(year0,yearT,5), "mean"),
                        list("Primary_Enrollmentratio" ,   seq(year0,yearT,5), "mean"),
                        list("Secondary_Enrollmentratio" ,  seq(year0,yearT,5), "mean"),
                        list("Tertiary_Enrollmentratio" ,  seq(year0,yearT,5), "mean"),
                        
                       # list("gdppercapita1990gkusd" ,  seq(1900,1938), "mean"),
                        list("Populationgrowth" ,   seq(1900,1938), "mean")
                      ),
                      dependent = "LNgdppc",
                      #dependent = "gdppercapita1990gkusd",
                      unit.variable = "country_number",
                      unit.names.variable = "country",
                      time.variable = "year",
                      treatment.identifier = 0,
                      #controls.identifier = c(1:2, 6, 14, 20:28),
                      controls.identifier = c(1:2, 4, 6, 8, 14, 18, 20, 22:28),
                      
                      #controls.identifier = c(1:2, 20:28),
                      time.optimize.ssr = 1900:1938,
                      time.plot = 1900:1960
             )

write.table(dataprep.out$X0, "X0.txt", sep="\t")
dataprep.out$Y0plot
names(AUX)
synth.out <- synth(data.prep.obj = dataprep.out,
                   method = "BFGS")

gaps <- dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w)


synth.tables <- synth.tab(dataprep.res = dataprep.out,
                          synth.res = synth.out)


synth.tables$tab.pred[1:5, ]

synth.tables$tab.pred
synth.tables$tab.v
synth.tables$tab.w
synth.tables$tab.loss

synthChileGdppc <- exp(dataprep.out$Y0plot%*%synth.out$solution.w)
chileGdppc <- exp(dataprep.out$Y1plot)
comparedChile <- cbind(synthChileGdppc, chileGdppc)
comparedChile


#RMSPE is:

RMSPE_0 <- mean((gaps^2)[1:24])^0.5

#RMSPE-to-mean ratio:

(mean((gaps^2)[1:24])^0.5 / mean(dataprep.out$Y1plot[1:24]))

#RMSPE ratio: (post/pre intervention)

RMSPE_1 <- mean((gaps^2)[25:length(gaps)])^0.5
RMSPE_0
RMSPE_1/RMSPE_0

path.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          Ylab = "Log real per-capita GDP (1990 USD)",
          Xlab = "Year",
          Ylim = c(7.4,9),
          Legend = c("Chile","Synthetic Chile"),
          Legend.position = "bottomright"
)
abline(v=1939,lty="dotted",lwd=2)
help(dataprep)
###################################################
### chunk number 14: 
###################################################
gaps.plot(synth.res = synth.out,
          dataprep.res = dataprep.out,
          Ylab = "gap in real per-capita GDP (1986 USD, thousand)",
          Xlab = "year",
          #Ylim = c(-1.5,1.5),
          Main = NA
)
abline(v=1924,lty="dotted",lwd=2)



##################################
## Setup up coordinate system (with x == y aspect ratio):
plot(c(-2,3), c(-1,5), type = "n", xlab = "x", ylab = "y", asp = 1)
## the x- and y-axis, and an integer grid
abline(h = 0, v = 0, col = "gray60")
text(1,0, "abline( h = 0 )", col = "gray60", adj = c(0, -.1))
abline(h = -1:5, v = -2:3, col = "lightgray", lty = 3)
abline(a = 1, b = 2, col = 2)
text(1,3, "abline( 1, 2 )", col = 2, adj = c(-.1, -.1))

#data visualization 
#first we want to show plot of Chile gdp vs LA average and USA-Canada

plot(1900:1950, exp(dataprep.out$Y1plot), type = "l")
dataprep.out$Y0plot

LAgdppc <- dataprep.out$Y0plot[,1:9]
LAgdppc <- (rowMeans(LAgdppc))

ARG <- dataprep.out$Y0plot[,1]
URU <- dataprep.out$Y0plot[,8]
gapChileARG <- (dataprep.out$Y1plot - ARG)^2
gapChileURU <- (dataprep.out$Y1plot - URU)^2
mean(gapChileARG[1:24])^0.5
mean(gapChileURU[1:24])^0.5

UsaCanada <- cbind(dataprep.out$Y0plot[,11], dataprep.out$Y0plot[,15])
cbind(1:5, 1:5)
UsaCanada <- rowMeans(UsaCanada)
UsaCanada
gapChileUsaCanada <- (dataprep.out$Y1plot - UsaCanada)^2

gapChileLA <- (dataprep.out$Y1plot - LAgdppc)^2
LAgdppc
mean(gapChileLA[1:24])^0.5
mean(gapChileUsaCanada[1:24])^0.5
RMSPE_0 <- mean((gaps^2)[1:24])^0.5

matplot(1900:1950, cbind(exp(dataprep.out$Y1plot), exp(LAgdppc), UsaCanada), type = "l")
plot(1900:1950, exp(dataprep.out$Y1plot),ylim = c(1000,10000), 
     xlab="Year", ylab="Real GDP per capita (1990 G-K$)", 
     xlim = c(1905, 1960), type = "l", lwd = 3)
lines(1900:1950, exp(dataprep.out$Y1plot), col ="black", lwd = 3)
lines(1900:1950, exp(LAgdppc), col = "gray", lwd = 3)
lines(1900:1950, UsaCanada, col = "purple", lwd = 3)
abline(v=1950,lty="dotted",lwd=2)
text(locator(), labels = c("Chile", "LA avg \n 0.6362173", "Usa-Canada avg \n (0.5805108)"))

matplot(1900:1950, cbind(dataprep.out$Y1plot, LAgdppc, UsaCanada), type = "l")
plot(1900:1950, dataprep.out$Y1plot, 
     xlab="Year", ylab="Log real GDP per capita (1990 USD)", ylim = c(7,9.5),
     xlim = c(1905, 1960), type = "l", lwd = 3)
lines(1900:1950, dataprep.out$Y1plot, col ="black", lwd = 3)
lines(1900:1950, LAgdppc, col = "gray", lwd = 3)
lines(1900:1950, UsaCanada, col = "purple", lwd = 3)
abline(v=1950,lty="dotted",lwd=2)
text(locator(), labels = c("Chile", "LA avg \n (0.636)", "Usa-Canada avg \n (0.581)"))
#0.6362173  0.5805108

matplot(1900:1950, cbind(dataprep.out$Y1plot, ARG, URU), type = "l")
plot(1900:1950, dataprep.out$Y1plot, 
     xlab="Year", ylab="Log real GDP per capita (1990 USD)", ylim = c(7.4,8.7),
     xlim = c(1905, 1960), type = "l", lwd = 3)
lines(1900:1950, dataprep.out$Y1plot, col ="black", lwd = 3)
lines(1900:1950, ARG, col = "gray", lwd = 3)
lines(1900:1950, URU, col = "purple", lty = "dashed",lwd = 3)
abline(v=1950,lty="dotted",lwd=2)
text(locator(), labels = c("Chile", "Argentina \n (0.369)", "Uruguay \n (0.122)"))
#0.369205  0.1222677



plot(1900:1950, chileGdppc, ylim = c(1500,6500), 
     xlab="Year", ylab="Real GDP per capita (1990 USD)", 
     xlim = c(1905, 1945), type = "l")
lines(1900:1950, chileGdppc, col ="black", lwd=2)
lines(1900:1950, synthChileGdppc, type = "l", lty = "dotted", lwd=2)
lines(1900:1950, results_matlab$Chile[1:51], type = "l", lty = "dashed", lwd=2)
abline(v=1924,lty="dotted",lwd=2)

results_matlab$Chile
legend("topleft", c("Chile", "Synthetic Chile Method 2", "Synthetic Chile Method 1"), lty=c(1,2,3), lwd=c(2.5,2.5),col=c("black","black"))
help(legend)
help(plot)

plot(1900:1950, chileGdppc, ylim = c(1500,6500), 
     xlab="Year", ylab="Real GDP per capita (1990 USD)", 
     xlim = c(1905, 1945), type = "l")
lines(1900:1950, chileGdppc, col ="black", lwd=2)
lines(1900:1950, synthChileGdppc, type = "l", lty = "dashed", lwd=2)
abline(v=1924,lty="dotted",lwd=2)

legend("bottomright", c("Chile", "Synthetic Chile"), lty=c(1,2), lwd=c(2.5,2.5),col=c("black","black"))
