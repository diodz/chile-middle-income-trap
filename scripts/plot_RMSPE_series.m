
function plot_RMSPE_series(year, ratio, figure_filename)

%Plots the RMSPE ratio

fig = plot(year, ratio, 'Color', [0, 0, 0]);
 
axis([1920 1955 -1 12]);

%title([treatmentCountryName, ' RMSPE ratio versus T0']);
xlabel('Year');
ylabel('RMSPE ratio');

legend('RMSPE Ratio', 'Location', 'northwest');
set(fig, 'LineWidth', 2);
set(gca,'fontsize',15)

print([figure_filename], '-dpng');
movefile([figure_filename, '.png'], 'Figures')
close;