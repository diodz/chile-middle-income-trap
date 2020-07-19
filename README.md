# Another case of the middle-income trap. Chile, 1900-1939
 
 Source code for article to be published in Revista de Historia Económica: 
  
 Correct citation to be updated when published.
 
 For the moment cite as: 
 
**Authors: Cristián Larroulet, Juan Pablo Couyoumdjian, Diego A. Díaz**

## Instructions

See first **main_markdown.Rmd** or **main_markdown.pdf** to produce figures 1 to 3 and instructions for figures 4 and 5.

Comparisons of relative income and structural break analysis is done in R in the files **data_preparation.R** and **structural_break_tests.R**. This code contains functions to produce the results from figures 1 and 2 by loading an unmodified version of Maddison (2018) data.

The file **prepare_data_for_matlab.R** processes Maddisson (2018) data by selecting the countries used to apply the synthetic control method in MATLAB, it also extracts the corresponding period used, from 1900 to 1960. The .csv files created by this script are **gdppc.csv** and **countries.csv**.

In MATLAB, see and run file **main_synthetic_controls.m** to produce figures 3, 4 and 5.
