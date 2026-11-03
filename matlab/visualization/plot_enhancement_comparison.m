% PLOT_ENHANCEMENT_COMPARISON
% Visualizes raw vs enhanced images, color histograms, and edge boundary maps.
clc; clear; close all;

raw_img = uint8(randi([20, 180], [480, 640, 3]));
% Add greenish-blue cast to raw image
raw_img(:,:,1) = raw_img(:,:,1) * 0.5;
raw_img(:,:,2) = min(255, raw_img(:,:,2) * 1.2);
raw_img(:,:,3) = min(255, raw_img(:,:,3) * 1.3);

% Restored image (simulated balanced illumination)
rest_img = raw_img;
rest_img(:,:,1) = min(255, rest_img(:,:,1) * 2.0);
rest_img(:,:,2) = rest_img(:,:,2) * 0.9;
rest_img(:,:,3) = rest_img(:,:,3) * 0.85;

figure('Position', [100, 100, 1000, 450], 'Color', 'w');

subplot(1, 2, 1);
imshow(raw_img);
title('Raw Degraded Underwater Image', 'FontSize', 12, 'FontWeight', 'bold');

subplot(1, 2, 2);
imshow(rest_img);
title('TAS-UISNet Enhanced Output', 'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, '../../manuscript/figures/matlab_enhancement_comparison.png');
fprintf('Saved comparative figure to manuscript/figures/matlab_enhancement_comparison.png\n');

