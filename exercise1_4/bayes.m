clear all; close all; clc;
%Natural parameters
mu1 = [2; 3];
sigma1 = [2 0.5; 0.5 1];
mu2 = [4; 4];
sigma2 = [1.5 -0.3; -0.3 0.8];

%Gaussians as anonymous functions 
gauss1 = @(x1, x2) exp(-0.5*([x1;x2]-mu1)'*sigma1*([x1;x2]-mu1))/(2*pi*sqrt(det(sigma1)));
gauss2 = @(x1, x2) exp(-0.5*([x1;x2]-mu2)'*sigma2*([x1;x2]-mu2))/(2*pi*sqrt(det(sigma2)));

[x1, x2] = meshgrid(-5:0.2:10); %generate grid
g1 = arrayfun(gauss1, x1, x2); %apply functions to grid points
g2 = arrayfun(gauss2, x1, x2);

figure;
for p1 = [.1, .25, .5, .75, .9]
    p2 = 1 - p1;
    
    surf(x1, x2, g1, 'FaceColor', [0.9 0.2 0.2]);  %plot
    hold on
    surf(x1, x2, g2, 'FaceColor', [0.2 0.9 0.2]);
    
    gdiff = p1 * g1 - p2 * g2; %get difference points
    
    C = contours(x1, x2, gdiff, [0,0]); %get points where contour is zero
    
    %get seperate coordinates of contour
    my_range = 2:107;
    x1s = C(1, my_range);
    x2s = C(2, my_range);

    % Fit a quadratic polynomial to the data
    p = polyfit(x1s, x2s, 2);
    
    % Print the resulting polynomial equation
    fprintf('(p1 = %.2f) The fitted function is: %.2fx^2 + %.2fx + %.2f\n', p1, p(1), p(2), p(3));
    
    %interporlate contour on gaussian 1
    boundry = interp2(x1, x2, g1, x1s, x2s);
    
    %plot it
    line(x1s, x2s, boundry, 'Color', 'k', 'LineWidth', 2);
end

fprintf("\nProgram paused. Press enter to continue.\n\n")
pause;
clear all;

%Natural parameters
sigma = [1.2 .4; .4 1.2];
mu1 = [2; 3];
sigma1 = sigma;
mu2 = [4; 4];
sigma2 = sigma;

%Gaussians as anonymous functions 
gauss1 = @(x1, x2) exp(-0.5*([x1;x2]-mu1)'*sigma1*([x1;x2]-mu1))/(2*pi*sqrt(det(sigma1)));
gauss2 = @(x1, x2) exp(-0.5*([x1;x2]-mu2)'*sigma2*([x1;x2]-mu2))/(2*pi*sqrt(det(sigma2)));

[x1, x2] = meshgrid(-5:0.2:10); %generate grid
g1 = arrayfun(gauss1, x1, x2); %apply functions to grid points
g2 = arrayfun(gauss2, x1, x2);

figure;
for p1 = [.1, .25, .5, .75, .9]
    p2 = 1 - p1;
    
    surf(x1, x2, g1, 'FaceColor', [0.9 0.2 0.2]);  %plot
    hold on
    surf(x1, x2, g2, 'FaceColor', [0.2 0.9 0.2]);
    
    gdiff = p1 * g1 - p2 * g2; %get difference points
    
    C = contours(x1, x2, gdiff, [0,0]); %get points where contour is zero
    
    %get seperate coordinates of contour
    x1s = C(1, 2:end);
    x2s = C(2, 2:end);
    
    % Fit a quadratic polynomial to the data
    p = polyfit(x1s, x2s, 1);
    
    % Print the resulting polynomial equation
    fprintf('(p1 = %.2f) The fitted function is: %.2fx + %.2f\n', p1, p(1), p(2));

    %interporlate contour on gaussian 1
    boundry = interp2(x1, x2, g1, x1s, x2s);
    
    %plot it
    line(x1s, x2s, boundry, 'Color', 'k', 'LineWidth', 2);
end

%saving_figs('C:\\Users\\harilaos\\Desktop\\1\\SMAP-proj1\\exercise1_4\\images');
