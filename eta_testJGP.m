function [yp sp] = testJGP(model,xp)

% Predictive mean
ntest = size(xp,1); r = size(model.x,1); s = size(model.xs,1);
new_cross = [ones(ntest,r), ones(ntest,s)*model.eta];
K  = kernelmatrix('rbf',xp',[model.x; model.xs]',model.sigma);
yp = (K.*new_cross) * model.alpha;

% Predictive variance = confidence intervals
Ktt   = kernelmatrix('rbf',xp',xp',model.sigma);
ntest = size(xp,1);
for m = 1:ntest
    sp(m) = model.sigman2 + Ktt(m,m) - K(m,:)*(model.C\K(m,:)');  % k** - k*: * inv(K+lambda*I)*k:*
end
sp = sp(:);
