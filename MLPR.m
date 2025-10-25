% Bhattacharyya Distance between two Gaussian distributions
clear; clc; close all;

% Define two Gaussian distributions
mu1 = 0; sigma1 = 1; % Mean and std of P
mu2 = 2; sigma2 = 1.5; % Mean and std of Q

% Generate x values
x = linspace(-5, 7, 1000);

% Probability Density Functions (PDFs)
P = normpdf(x, mu1, sigma1);
Q = normpdf(x, mu2, sigma2);

% Bhattacharyya Coefficient (BC) - Integral of sqrt(P*Q)
BC = trapz(x, sqrt(P .* Q));

% Bhattacharyya Distance (DB)
DB = -log(BC);

fprintf('Bhattacharyya Coefficient (BC) = %.4f\n', BC);
fprintf('Bhattacharyya Distance (DB) = %.4f\n', DB);

% Plot the distributions and their overlap
figure;
plot(x, P, 'b-', 'LineWidth', 2, 'DisplayName', 'P ~ N(\mu_1, \sigma_1^2)');
hold on;
plot(x, Q, 'r-', 'LineWidth', 2, 'DisplayName', 'Q ~ N(\mu_2, \sigma_2^2)');
fill(x, min(P, Q), 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Overlap (BC)');

xlabel('x');
ylabel('Probability Density');
title(['Bhattacharyya Distance: D_B = ', num2str(DB, '%.4f')]);
legend('show');
grid on;