
close all; clear all; clc; 

[LatNE, LonNE] =  deal(25.928337, -80.31182);
[LatSE, LonSE] =  deal(25.932055, -80.31258);
[LatNW, LonNW] =  deal(25.928507, -80.31079);
[LatSW, LonSW] =  deal(25.932222, -80.311554);

% Extract latitude, longitude, and radius columns

originalString = 'preprocessed_output_%d_random.csv';
targetstring='lat_lon_%d.pdf';

for confidence=[90,95,99]
    figure;
    % Format the string with the number using sprintf
    updatedString = sprintf(originalString, confidence);

    data = readtable(updatedString, 'Format','%f%s%f%f%f%f');

    data.start_anomaly = datetime(data.start_anomaly,'inputFormat','yyyy-MM-dd HH:mm:ss','PivotYear',2000);
    r = 0.00002;

    latitudes = data.lat;
    longitudes = data.lon;
    radii = data.anomalySampleLength;
    dates = data.start_anomaly;
    dateMin = min(dates);
    dateMax = max(dates);

    maxWindowSize = max(radii);
    minWindowSize = min(radii);
    radii = (data.anomalySampleLength - minWindowSize) / (maxWindowSize - minWindowSize) +1;

    lat0 = 25.930337;
    lon0 = -80.31208;

    [lat,lon] = scircle1(lat0,lon0,r);

    % Increase the figure size
    figureSize = [100, 100, 1600, 1200];  % [left, bottom, width, height]
    set(gcf, 'Position', figureSize);

    geobasemap openStreetMap
    hold on;

    % geoplot(lat,lon,"k","LineWidth",2, "Color", "blue")
    % geoplot([LatNW LatNE LatSE LatSW LatNW], [LonNW LonNE LonSE LonSW LonNW], LineWidth = 2); 

    A=15;
    %%% pick the first anomaly and plot for legend purposes
    indices = find(radii>0);
    % [lat,lon] = scircle1(latitudes(indices(1)),longitudes(indices(1)),r);
    geoscatter(latitudes(indices(1)),longitudes(indices(1)), A, "red", "filled")

    colorScale = autumn(100);

    for index = 1:numel(latitudes)
        [lat,lon] = scircle1(latitudes(index),longitudes(index),r);

        if radii(index) > 1
            date = dates(index); 
            colorIndex = round(interp1(linspace(datenum(dateMin), datenum(dateMax), size(colorScale, 1)), 1:size(colorScale, 1), datenum(date)));
            color = colorScale(colorIndex, :);
            
            geoscatter(latitudes(index),longitudes(index), radii(index)*3*A, color, "filled")
        else
            geoscatter(latitudes(index),longitudes(index), 1.5*A, "green", "filled")
        end
    end
    legend("anomaly", "no anomaly")
    colormap(colorScale);
    c = colorbar;
    c.Label.String = 'Date';
    caxis([1, 10])
    c.Ticks = 1:10; % Customize the number of ticks as needed
    c.TickLabels ={'2017-08-29', '2017-10-24','2017-12-19','2018-02-13','2018-04-10','2018-06-05','2018-07-31','2018-09-25', '2018-11-20', '2019-01-15'};
    % c.TickLabels = cellstr(datestr(c.Ticks, 'yyyy-mm-dd')); % Format the tick labels as desired

    ax = gca;
    updatedString = sprintf(targetstring, confidence);
    exportgraphics(ax, 'lat_lon_99.pdf', 'Resolution',350)

end


