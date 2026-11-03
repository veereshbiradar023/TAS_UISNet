% EVALUATE_ALL_METRICS Batch evaluation of underwater image quality and boundaries
% TAS-UISNet Evaluation Suite for Elsevier IVC

clc; clear; close all;

addpath('../enhancement_analysis');
addpath('../color_analysis');
addpath('../boundary_analysis');

test_dir = '../../data/URPC2020/images/val';
if ~exist(test_dir, 'dir')
    fprintf('Test directory not found, running demo verification test...\n');
    demo_img = uint8(randi([30, 220], [480, 640, 3]));
    [uiqm, uicm, uism, uiconm] = calculate_uiqm(demo_img);
    uciqe = calculate_uciqe(demo_img);
    stats = analyze_color_cast(demo_img);
    fprintf('=== Synthetic Demo Validation ===\n');
    fprintf('UIQM: %.4f | UCIQE: %.4f | Angular Error: %.2f deg\n', uiqm, uciqe, stats.angular_error_deg);
    return;
end

img_files = dir(fullfile(test_dir, '*.jpg'));
num_images = length(img_files);

uiqm_arr = zeros(num_images, 1);
uciqe_arr = zeros(num_images, 1);

fprintf('Evaluating %d images in %s...\n', num_images, test_dir);
for i = 1:num_images
    img_path = fullfile(test_dir, img_files(i).name);
    img = imread(img_path);
    uiqm_arr(i) = calculate_uiqm(img);
    uciqe_arr(i) = calculate_uciqe(img);
end

fprintf('\n=== Final Dataset Evaluation Summary ===\n');
fprintf('Mean UIQM:  %.4f +/- %.4f\n', mean(uiqm_arr), std(uiqm_arr));
fprintf('Mean UCIQE: %.4f +/- %.4f\n', mean(uciqe_arr), std(uciqe_arr));

