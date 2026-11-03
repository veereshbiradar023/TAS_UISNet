function stats = analyze_color_cast(img)
% ANALYZE_COLOR_CAST Evaluates color distribution balance and channel attenuation
if ischar(img) || isstring(img)
    img = imread(img);
end
img = double(img);

r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);

mean_r = mean(r(:));
mean_g = mean(g(:));
mean_b = mean(b(:));

total = mean_r + mean_g + mean_b + 1e-6;
ratio_r = mean_r / total;
ratio_g = mean_g / total;
ratio_b = mean_b / total;

% Angular error relative to neutral gray illuminant [1/3, 1/3, 1/3]
v_actual = [mean_r, mean_g, mean_b] / norm([mean_r, mean_g, mean_b]);
v_ideal = [1, 1, 1] / sqrt(3);
angular_error_deg = acosd(dot(v_actual, v_ideal));

stats = struct(...
    'mean_R', mean_r, ...
    'mean_G', mean_g, ...
    'mean_B', mean_b, ...
    'ratio_R', ratio_r, ...
    'ratio_G', ratio_g, ...
    'ratio_B', ratio_b, ...
    'angular_error_deg', angular_error_deg ...
);
end

