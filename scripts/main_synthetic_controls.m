%% Main --- Implements the Abadie synthetic control method with one variable
% Figures 3, 4 and 5
% By Diego A. Diaz (https://github.com/diodz/chile-middle-income-trap)

% Please see published version of the article for correct citation. If not
% available, cite as: 
% Couyoumdjian, JP., Larroulet, C., Diaz, D.A. (2020) Another case of the 
% middle-income trap: Chile, 1900-1939. Revista de Historia Economica. 

%% Load Data - country names and GDP per capita from Maddison

countries = table2array(readtable('../data/countries.csv'));
data = csvread('../data/gdppc.csv', 1, 0);

%% Figure 4_1: Synthetic control for Chile
% The number of the treated unit must be the column of the country that 
% had the treatment in the data. Chile = 2

treatedUnit = 2;
treatmentYear = 1939;
[~, ~, ~, W] = synthetic_control(treatedUnit, data, treatmentYear,...
countries, 'Figure 4_1');

%% Weights for sinthetic control

disp('The weights of the synthetic control are:')
disp(horzcat(countries, num2cell(round(W, 6))))
clearvars W

%% Placebos
%  We make all synths for placebo testing

[T, ~] = size(data);
treatmentYear = 1939;
periods_for_RMSPE = 5;

WEIGHTS = zeros(length(countries)+1, length(countries));
GAPS = zeros(T+1, length(countries)-1);
RMSPE = zeros(3, length(countries)-1);

for i = 1:(length(countries)-1)
    
    [RMSPEpre, RMSPEpost, gaps, W] = synthetic_control_no_plot(i+1, data,...
        treatmentYear, countries, periods_for_RMSPE);
    
    WEIGHTS(:,i) = W;
    
    GAPS(1,i) = i+1;
    GAPS(2:end,i) = gaps;
    
    RMSPE(1, i) = i+1;
    RMSPE(2, i) = RMSPEpre;
    RMSPE(3, i) = RMSPEpost;
    
end

%% Figure 4_2: Placebo test.
% To avoid saturating the graph, we don't show those units that have an 
% RMSPE higher that 1.5 times that of Chile

factor = 1.5;
RMSPEmin = RMSPE(2,(treatedUnit-1))*factor;
treatmentCountryName = char(countries(treatedUnit-1));
placebo_plot_clean(GAPS, treatedUnit, treatmentYear, 1900,...
    treatmentCountryName, RMSPE, RMSPEmin, 'Figure 4_2');
clearvars factor RMSPEmin gaps GAPS i periods_for_RMSPE RMSPEpre RMSPEpost...
    T W WEIGHTS

%% Synthetic controls between 1925-1950
% Since we change the start year in this section, we need to change the
% before and after period considered. We choose 5 years.

periods_for_RMSPE = 5;
[T, variables] = size(data);
WEIGHTS = zeros(variables, 22);
GAPS = zeros(T, 22);
RMSPE = zeros(2, 22);

for t = 1:27
    [RMSPEpre, RMSPEpost, gaps, W] = synthetic_control_no_plot(treatedUnit,...
        data, 1923+t, countries, periods_for_RMSPE);
    WEIGHTS(:,t) = W;
    GAPS(:,t) = gaps;
    RMSPE(1, t) = RMSPEpre;
    RMSPE(2, t) = RMSPEpost;
end

%% Figure 3 RMSPE - Ratio

year = (1924:1950)';
ratio = RMSPE(2,:)./RMSPE(1,:);
plot_RMSPE_series(year, ratio, 'Figure 3');
clearvars RMSPE gaps GAPS t RMSPEpre RMSPEpost T W WEIGHTS variables ratio

%% Figure 5_1: Jackknife

data(:, 11) = [];
countries(10) = [];
[RMSPEpre, RMSPEpost, gaps, W] = synthetic_control(treatedUnit, data,...
    treatmentYear, countries, 'Figure 5_1');

%% Weights for Jackknife sinthetic control

disp('The weights of the synthetic control are:')
disp(horzcat(countries, num2cell(round(W, 6))))

%% Figure 5_2: Placebo test for jackknife model

[T, ~] = size(data);
treatmentYear = 1939;

WEIGHTS = zeros(length(countries)+1, length(countries));
GAPS = zeros(T+1, length(countries)-1);
RMSPE = zeros(3, length(countries)-1);

for i = 1:(length(countries)-1)
    
    [RMSPEpre, RMSPEpost, gaps, W] = synthetic_control_no_plot(i+1, data,...
        treatmentYear, countries, periods_for_RMSPE);
    
    WEIGHTS(:,i) = W;
    
    GAPS(1,i) = i+1;
    GAPS(2:end,i) = gaps;
    
    RMSPE(1, i) = i+1;
    RMSPE(2, i) = RMSPEpre;
    RMSPE(3, i) = RMSPEpost;
end

%% Figure 5_2: Plots placebo test

factor = 1.5;
RMSPEmin = RMSPE(2,(treatedUnit-1))*factor;
placebo_plot_clean(GAPS, treatedUnit, treatmentYear, 1900,...
    treatmentCountryName, RMSPE, RMSPEmin, 'Figure 5_2');
