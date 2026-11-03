function uciqe = calculate_uciqe(img)
% CALCULATE_UCIQE Computes Underwater Color Image Quality Evaluation
% Based on Yang and Xu, IEEE TIP.
% UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s

if ischar(img) || isstring(img)
    img = imread(img);
end

% Convert RGB to CIELab
lab = rgb2lab(img);
L = lab(:,:,1);
a = lab(:,:,2);
b = lab(:,:,3);

% Chroma std dev
chroma = sqrt(a.^2 + b.^2);
sigma_c = std(chroma(:));

% Luminance contrast (top 1% minus bottom 1%)
L_sorted = sort(L(:));
n = numel(L_sorted);
top_idx = round(0.99 * n);
bot_idx = round(0.01 * n);
con_l = L_sorted(top_idx) - L_sorted(bot_idx);

% Mean saturation in HSV
hsv = rgb2hsv(img);
mu_s = mean(hsv(:,:,2), 'all');

c1 = 0.4680;
c2 = 0.2745;
c3 = 0.2576;

uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s;
end

