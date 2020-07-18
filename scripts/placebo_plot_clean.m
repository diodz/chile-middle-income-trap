function placebo_plot_clean(gaps, treatmentCountry, treatmentYear,...
    initialYear, treatmentCountryName, RMSPE, minRMSPE, figure_filename)

[period, countries] = size(gaps);
finalYear = period + initialYear - 2;

Y1 = gaps(2:end, treatmentCountry-1);

for i = 1:countries
   if RMSPE(2, (countries-i+1)) > minRMSPE
       RMSPE(:, (countries-i+1)) = [];
       gaps(:, (countries-i+1)) = [];
   end
end

Y2 = gaps(2:end, :);
year = (initialYear:finalYear)';
placebos = plot(year, Y1, 'Color', [0, 0, 0]);
 
hold on
plot(year, Y2, 'Color', [17, 17, 17]/30)
plot(year, Y1, 'Color', [0, 0, 0]);
axis([1900 1960 -4000 4000])

l = line([treatmentYear treatmentYear], get(gca, 'YLim'), 'Color',...
    [0, 0, 0]);
fig = plot(get(gca,'xlim'), [0 0], 'Color', [0,0,0]);

%% Adjusting display and colors

%title([treatmentCountryName, ' placebo test ', num2str(treatmentYear)]);
xlabel('Year');
ylabel('gdppc');
set(l, 'LineStyle', '--');

legend(treatmentCountryName, 'Placebos', 'Location', 'northeast');
set(fig, 'LineWidth', 2);
set(gca,'fontsize',15);
placebos(1).LineWidth = 3;
print(figure_filename, '-dpng');
movefile([figure_filename, '.png'], '../figures');

close;
