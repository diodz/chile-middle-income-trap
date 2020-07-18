%% This synthetic control method allows for an intercept and negative
%weights.
%Current configuration imposes this restrictions, but they can be changed
%easily in the inequalities section

function [RMSPEpre, RMSPEpost, gaps, W] = synthetic_control_no_plot(...
    treatmentCountry, data, T0, countries, periods_for_RMSPE)

year = data(:, 1); 
[~, nVariables] = size(data);
nControls = nVariables - 2;
initialYear = 1900;
T0 = -initialYear+T0;
treatmentYear = initialYear + T0;

treatmentCountryName = char(countries(treatmentCountry-1));

Z1 = data(1:T0, treatmentCountry);
Z0 = data(1:T0, 2:nVariables); 

%We set an initial guess for our weights vector
W0 = ones(nVariables, 1)*(1/nControls);

%And for the intercept
W0(nControls+1) = 0; 
W0(1) = 0;

%%
%Linear inequalities
A = -eye(nVariables);
b = zeros(length(W0), 1)+0; %if 0 only allow positive weights
b(nVariables) = 0;

%%
%Linear equalities
adding_UP = 1; %to remove restriction ADDING UP, equal to 0

Aeq = ones(nVariables, 1)'*adding_UP; 
Aeq(treatmentCountry-1) = 0;
Aeq(nVariables) = 0;

A2eq = zeros(nVariables,1)'; %Intercept
A2eq(nVariables) = 1; %if 0 no restriction on intercept, if 1 then it's 0

A3eq = zeros(nVariables,1)'; %treatment country weight must be 0
A3eq(treatmentCountry-1) = 1;

beq = 1;%ones(nVariables, 1)*adding_UP;
b2eq = 0;
b3eq = 0;

Aeq = [A3eq; A2eq; Aeq];
beq = [b3eq; b2eq; beq];

%%
%Define function handle
fun = @(x)synth(x, Z1, Z0);
W = fmincon(fun, W0, A, b, Aeq, beq);

%% Plot

Y1 = data(:, treatmentCountry);
Y2 = data(:, 2:nVariables);
synthControl = Y2*W(1:(nControls+1));

fig = plot(year, Y1);
hold on
plot(year, synthControl, '--',  'LineWidth',3)
l = line([treatmentYear treatmentYear], get(gca, 'YLim'), 'Color', [0, 0, 0]);
set(l, 'LineStyle', '--');

%% Adjusting colors and display

title([treatmentCountryName, ' Synthetic Control ',num2str(treatmentYear)]);
xlabel('Year');
ylabel('gdppc');
axis([initialYear 1960 -Inf +Inf]);

legend(treatmentCountryName, 'Synthetic control', 'Location', 'northwest');
set(fig, 'LineWidth', 2);
set(gca,'fontsize',15)
close;
    
%% Storing more results

gaps = (Y1-Y2*W(1:(nControls+1)));
spe = gaps.^2;
RMSPEpre = mean(spe(T0-periods_for_RMSPE:T0))^0.5;
RMSPEpost = mean(spe((T0+1):T0+1+periods_for_RMSPE))^0.5;

