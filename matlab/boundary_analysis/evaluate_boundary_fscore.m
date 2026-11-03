function [F_B, precision, recall] = evaluate_boundary_fscore(pred_mask, gt_mask, bound_th)
% EVALUATE_BOUNDARY_FSCORE Computes Boundary F-score (F_B)
% Measures accuracy of predicted instance boundaries against ground truth contours
if nargin < 3
    bound_th = 2.0; % 2 pixel error tolerance
end

pred_b = bwperim(pred_mask > 0);
gt_b = bwperim(gt_mask > 0);

if ~any(pred_b(:)) && ~any(gt_b(:))
    F_B = 1.0; precision = 1.0; recall = 1.0;
    return;
elseif ~any(pred_b(:)) || ~any(gt_b(:))
    F_B = 0.0; precision = 0.0; recall = 0.0;
    return;
end

dist_gt = bwdist(gt_b);
dist_pred = bwdist(pred_b);

precision = sum(dist_gt(pred_b) <= bound_th) / sum(pred_b(:));
recall = sum(dist_pred(gt_b) <= bound_th) / sum(gt_b(:));

if (precision + recall) == 0
    F_B = 0.0;
else
    F_B = 2.0 * precision * recall / (precision + recall);
end
end

