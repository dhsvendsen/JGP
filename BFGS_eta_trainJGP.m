function model = BFGS_trainJGP(xtr,ytr,xs,ys,xts,yts)

X_all=[xtr;xs];
y_all=[ytr;ys];
r=size(xtr,1);
s=size(xs,1); 

% theta = [sigma sigman2 gamma]
jgp_nll_fixdata = @(theta) jgp_nll(theta, X_all, y_all, r, s);
theta_0 = [median(pdist(xtr)) ; sqrt(var(diff(ytr))); 1 ; 0];
theta_0 = good_init(theta_0, jgp_nll_fixdata);

lowerbound = [theta_0(1)*0.5;0;0;0]; upperbound = [10*theta_0(1:3);100];
options = optimoptions('fmincon','GradObj','on');
model.opt = fmincon(jgp_nll_fixdata,theta_0, [],[],[],[], lowerbound, upperbound, []);%, options);

% Final model
model.sigma = model.opt(1);
model.sigman2 = model.opt(2);
model.gamma = model.opt(3);
model.eta = model.opt(4);
model.x = xtr;
model.xs = xs;
model.y = ytr;
model.ys = ys;

Knn = kernelmatrix('rbf', [xtr;xs]', [xtr;xs]', model.sigma);
M_cross = [ones(r),model.eta*ones(r,s); model.eta*ones(s,r),ones(s)]; 
model.C = (Knn.*M_cross + diag( [ones(1,r), ones(1,s)/ model.gamma] )*model.sigman2 );
model.alpha = model.C \ [ytr;ys];

%compute TEST-set RMSE
ntest = size(xts,1);
Knew = kernelmatrix('rbf', xts', [xtr;xs]', model.sigma);
new_cross = [ones(ntest,r), ones(ntest,s)*model.eta];
yp = (Knew.*new_cross)*model.alpha;
model.res = sqrt( mean((yp-yts).^2) );
model.res2 = mean(abs(yts-yp));
end

function f = jgp_nll(theta, X_all, y_all, r, s)

% jgp_nll.m This function returns the function value, partial derivatives
% of the negative loglikelihood funtion of the JGP given by:
%
%       f(theta) = - (-0.5* ln( |C_N| ) - 0.5* y^T(C_N^-1)y - konst )
%       where
%       C_N = kernelmatrix('rbf',X_n,X_n,theta_1) + diag( theta_2*ones(1,r), (theta_2/theta_3)*ones(1,s) )
%       where r and s are the number of real/synth data points respectively
%

% Get variables from BFGS_trainJGP


% compute negative log likelihood function
Kvar = diag( [ones(1,r), ones(1,s) / theta(3) ] )*theta(2);
Knn = kernelmatrix('rbf', X_all', X_all', theta(1));
M_cross = [ones(r),theta(4)*ones(r,s); theta(4)*ones(s,r),ones(s)]; 
C_N = (Knn.*M_cross + Kvar);% + cross);


if min(eig(C_N)) < 0
    'bad inverse matrix!'
    f = Inf
else
    f = log(det(C_N)) + y_all'*(C_N\y_all) ;

end

%if nargout > 1
%    dCN_dtheta1 = D/(4*theta(2)^3).*Knn;
%    dCN_dtheta2 = [ones(1,r), ones(1,s) / theta(3) ];
%    dCN_dtheta3 = -[zeros(1,r), ones(1,s) / (theta(3)^2) ]*theta(2);


%    df_dtheta1 = 0.5*trace(C_N\dCN_dtheta1) - 0.5*y_all'*(C_N\dCN_dtheta1)*(C_N\y_all) ;
%    df_dtheta2 = 0.5*trace(C_N\dCN_dtheta1) - 0.5*y_all'*(C_N\dCN_dtheta1)*(C_N\y_all) ;
%    df_dtheta3 = 0.5*trace(C_N\dCN_dtheta1) - 0.5*y_all'*(C_N\dCN_dtheta1)*(C_N\y_all) ;
%    df = [df_dtheta1 ; df_dtheta2 ; df_dtheta3];
%end

end


function init = good_init(theta_0,fn)

'seeking alternative init'

init = theta_0;

if ~isnan(fn(theta_0)) & ~isinf(fn(theta_0))
    % if the initialization is good from the beginning dont touch
    'the given init was good'
    return
else
    % if it isn't good continue searching randomly
    it = 1
    while isnan(fn(theta_0)) || isinf(fn(theta_0))
        theta_0(1) = init(1)*rand(1)*10;
        theta_0(2) = init(2)*rand(1)*10;
        it = it + 1;
        if it==100
            % give up after 100 tries
            'no good init was found'
            return
        end
    end
    'good new init was found'
    init = theta_0;
end

end