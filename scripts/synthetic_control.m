%% Implements the synthetic control method to reproduce the results in 
%  Another case of the middle-income trap: Chile, 1900-1939

function [RMSPEpre, RMSPEpost, gaps, W] = synthetic_control(...
    treatmentCountry, data, T0, countries, figure_filename)

year = data(:, 1); 
[~, nVariables] = size(data);
nControls = nVariables - 2;
initialYear = 1900;
T0 = -initialYear + T0;
treatmentYear = initialYear + T0;

treatmentCountryName = char(countries(treatmentCountry-1));

Z1 = data(1:T0, treatmentCountry);
Z0 = data(1:T0, 2:nVariables); 

% We set an initial guess for our weights vector
W0 = ones(nVariables, 1)*(1/nControls);

% We set a guess for the intercept
W0(nControls+1) = 0; 
W0(1) = 0;

%% Linear inequalities

A = -eye(nVariables);
b = zeros(length(W0), 1)+0; %if 0 only allow positive weights
b(nVariables) = 0;

%% Linear equalities

% As default we have an adding-up restriction, which means that synthetic 
% control weights must be equal to 1. To remove this restriction we simply
% have to make adding_UP equal to 0.

adding_UP = 1;

Aeq = ones(nVariables, 1)'*adding_UP; 
Aeq(treatmentCountry-1) = 0;
Aeq(nVariables) = 0;

% Intercept restriction:
% If 0 no restriction on intercept, if 1 we don't have an intercept
A2eq = zeros(nVariables,1)';
A2eq(nVariables) = 1; 

% Treatment country weight must be 0
A3eq = zeros(nVariables,1)'; 
A3eq(treatmentCountry-1) = 1;

beq = 1;
b2eq = 0;
b3eq = 0;

Aeq = [A3eq; A2eq; Aeq];
beq = [b3eq; b2eq; beq];

%% We define function handle in a separate function and find the minimal weights

fun = @(x)synth(x, Z1, Z0);
W = fmincon(fun, W0, A, b, Aeq, beq);

% Since the last entry is the value of the intercept, we don't include it
% in the output

W = W(1 : end-1);

%% Plot

Y1 = data(:, treatmentCountry);
Y2 = data(:, 2:nVariables);
synthControl = Y2*W(1:(nControls+1));

fig = plot(year, Y1, 'Color', [0, 0, 0]);
hold on
plot(year, synthControl, '--',  'LineWidth', 1, 'Color', [1, 0, 0])
l = line([treatmentYear treatmentYear], get(gca, 'YLim'), 'Color',...
    [0, 0, 0]);
set(l, 'LineStyle', '--');

%% Adjusting colors and display

%title([treatmentCountryName, ' Synthetic Control ', num2str(treatmentYear)]);
xlabel('Year');
ylabel('Real GDP per capita [2011 US$]');
axis([initialYear 1960 -Inf +Inf]);

legend(treatmentCountryName, 'Synthetic control', 'Location', 'northwest');
set(fig, 'LineWidth', 1);
set(gca,'fontsize',12)
print(figure_filename, '-dpng');
movefile([figure_filename, '.png'], '../figures')
close;
    
%% More results: Gaps in outcome variable and RMSPE before and after 

gaps = (Y1-Y2*W(1:(nControls+1)));
spe = gaps.^2;
RMSPEpre = mean(spe(1:T0))^0.5;
RMSPEpost = mean(spe((T0+1):end))^0.5;

