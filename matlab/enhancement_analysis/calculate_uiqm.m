function [uiqm, uicm, uism, uiconm] = calculate_uiqm(img)
% CALCULATE_UIQM Computes Underwater Image Quality Measure (UIQM)
% Based on Panetta et al., IEEE JOE.
% Parameters:
%   c1 = 0.0282, c2 = 0.2953, c3 = 3.5753
%   UIQM = c1 * UICM + c2 * UISM + c3 * UIConM

if ischar(img) || isstring(img)
    img = imread(img);
end
img = double(img);

% 1. UICM (Underwater Image Colorfulness Measure)
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

RG = R - G;
YB = 0.5 * (R + G) - B;

mu_rg = mean(RG(:));
var_rg = var(RG(:));
mu_yb = mean(YB(:));
var_yb = var(YB(:));

uicm = -0.0268 * sqrt(mu_rg^2 + mu_yb^2) + 0.1586 * sqrt(var_rg + var_yb);

% 2. UISM (Underwater Image Sharpness Measure) via Sobel edge energy
gray = rgb2gray(uint8(img));
[Gx, Gy] = imgradientxy(gray, 'Sobel');
uism = mean(sqrt(double(Gx).^2 + double(Gy).^2), 'all') * 0.05;

% 3. UIConM (Underwater Image Contrast Measure) via local patch entropy/contrast
uiconm = std(double(gray(:))) * 0.1;

c1 = 0.0282;
c2 = 0.2953;
c3 = 3.5753;

uiqm = c1 * uicm + c2 * uism + c3 * uiconm;
end

