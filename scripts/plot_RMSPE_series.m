
function plot_RMSPE_series(year, ratio, figure_filename)

%Plots the RMSPE ratio

fig = plot(year, ratio, 'Color', [0, 0, 0]);

axis([1920 1955 -1 12]);

%title([treatmentCountryName, ' RMSPE ratio versus T0']);
xlabel('Year');
ylabel('RMSPE ratio');

l = line([1941 1941], get(gca, 'YLim'), 'Color', [0, 0, 0]);
set(l, 'LineStyle', '--');

%legend('RMSPE Ratio', 'Location', 'northwest');
set(fig, 'LineWidth', 1);
set(gca,'fontsize',12)

print([figure_filename], '-dpng');
movefile([figure_filename, '.png'], '../figures')
close;