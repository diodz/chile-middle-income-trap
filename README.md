# Another case of the middle-income trap: Chile, 1900-1939

## Forthcoming: Revista de Historia Económica

**Authors: Cristián Larroulet, Juan Pablo Couyoumdjian, Diego A. Díaz**
 
 Source code for article.
  
 Correct citation to be updated when published. 

## Instructions

See and run file **main_markdown.Rmd** to produce figures 1 to 3 and instructions for figures 4 and 5. This code produces the pdf **main_markdown.pdf** and calls the functions required to produce the figures in the paper.

Comparisons of relative income and structural break analysis are done in R in the files **data_preparation.R** and **structural_break_tests.R**. These scripts contain functions to produce the results from figures 1 and 2 by loading an unmodified version of Maddison (2018) data.

The file **prepare_data_for_matlab.R** processes Maddisson (2018) data by selecting the countries used to apply the synthetic control method in MATLAB, it also extracts the corresponding period used, from 1900 to 1960. The .csv files created by this script are **gdppc.csv** and **countries.csv**.

In MATLAB, run file **main_synthetic_controls.m** to produce figures 3, 4 and 5.

## Data

All data used comes from Maddison (2018) and it's stored in **data/mpd2018.xlsx**. It can also be downloaded from the original website: https://www.rug.nl/ggdc/historicaldevelopment/maddison/data/mpd2018.xlsx. 

Any use of this data should use the following reference: 

Maddison Project Database, version 2018. Bolt, Jutta, Robert Inklaar, Herman de Jong and Jan Luiten van Zanden (2018), “Rebasing ‘Maddison’: new income comparisons and the shape of long-run economic development”, Maddison Project Working paper 10
