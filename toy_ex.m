addpath(genpath(pwd))
clear all, close all

%% SETUP of FIGURES
fontname = 'AvantGarde';
fontsize = 13;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',2,'DefaultLineMarkerSize',50,'DefaultLineColor',[0 0 0]);


%% Toy data
N = 20;     % #real data for training
M = 40;    % #synthetic data for training
ratio = 100*(1 - N/M)
Q = 500;    % #data for testing
% Frequency
freq  = 1; w = 2*pi*freq;
%% Training data
% Real noisy data (acquired in situ) for training
std_tr = 0.3;
xtr = linspace(-0.6,0.4,N)'; ytr = exp(-xtr).*sin(w*xtr) + std_tr*randn(N,1);
%% Synthetic NON-clean data (generated by an RTM)
std_sim = 0.2;
xxtr = linspace(-1,1,M)'; yytr = 1+exp(-xxtr).*sin(w*xxtr) + std_sim*randn(M,1);
%xxtr = [linspace(-1,-0.6,round(M/2)) , linspace(0.4,1,round(M/2)) ]'; yytr = exp(-xxtr).*sin(w*xxtr) + std_sim*randn(M,1);
%% Testing data: Real noisy data (acquired in situ)
std_ts = 0.1;
xts = linspace(-1,1,Q)'; yts = exp(-xts).*sin(w*xts) ;%+ std_ts*randn(Q,1);

X_all = [xtr;xxtr]; Y_all = [ytr;yytr];



%% fitting a regular GP
%  mean, covariance and likelihood functions
meanfunc_gp = []; % empty: don't use a mean function
covfunc_jgp = @covSEisoU; % Squared Exponential covariance function plus noise on simulated data
likfunc = @likGauss; % Gaussian likelihood

% Initialize the hyperparameter struct
hyp_init_gp.mean = [];
ell = 0.1; hyp_init_gp.cov = [log(ell)];
sn = 0.1; hyp_init_gp.lik = log(sn);

% Fit hyperparams
inf = @infGaussLik;
hyp_gp = minimize(hyp_init_gp, @gp, -100, inf, meanfunc_gp, covfunc_jgp, likfunc, X_all, Y_all);
% predict
[mu_gp s2_gp] = gp(hyp_gp, @infGaussLik, meanfunc_gp, covfunc_jgp, likfunc,  X_all, Y_all, xts);


%% fitting the JGP with cross term
cross_jgp = BFGS_eta_trainJGP(xtr,ytr,xxtr,yytr,xts,yts) % obs! the training set is used for development (nice) and has cross term
[mu_djgp s2_djgp]   = eta_testJGP(cross_jgp,xts);


%% plot figure 1
figure, 
     %f  = [mu_gp+2*sqrt(s2_gp); flipdim(mu_gp-2*sqrt(s2_gp),1)]; fill([xts; flipdim(xts,1)], f, [5 5 5]/8)
     hold on   
     %f2 = [mu_jgp+2*sqrt(s2_jgp); flipdim(mu_jgp-2*sqrt(s2_jgp),1)]; fill([xts; flipdim(xts,1)], f2, [7 7 7]/8)
     %f  = [mu_gp+2*sqrt(s2_gp); flipdim(mu_gp-2*sqrt(s2_gp),1)]; fill([xts; flipdim(xts,1)], f, [5 5 5]/8)
     %f3 = [mu_djgp+2*sqrt(s2_djgp); flipdim(mu_djgp-2*sqrt(s2_djgp),1)]; fill([xts; flipdim(xts,1)], f3, [7.5 7 8]/8, 'DisplayName','dJGP')
     plot(xts,yts,'k','DisplayName','True f'),
     plot(xts,mu_gp,'r','DisplayName','GP ')
     plot(xts,mu_djgp,'m','DisplayName','JGP')
     plot(xtr,ytr,'b.','markersize',20, 'DisplayName','Train_r'),
     plot(xxtr,yytr,'b+','markersize',7,'Linewidth',1.5,'DisplayName','Train_s'),
     legend('show')
     set(gca,'XMinorTick','on','YMinorTick','on'), grid
     xlabel('x'),ylabel('y')
     xlim([xts(2) xts(end-2)]);
     print -depsc2 figure1.eps

'The kernel-sigma and noise-sigma paramters of the regular GP (trained on both real and sim data) are:'
exp([hyp_gp.cov, hyp_gp.lik]) % sigma, noise

'The kernel-sigma, noise-sigma, gamma and eta paramters of the JGP are:'
[cross_jgp.sigma sqrt( cross_jgp.sigman2 ) cross_jgp.gamma cross_jgp.eta ]
'(gamma being higher than 1 means that there is less noise in the sim data. The smaller the eta the lower the similarity between real and sim data. )'

'The RMSE of the GP and JGp approaches are respectively'
cut=0%
[sqrt(mean( (mu_gp(1+cut:end-cut) - yts(1+cut:end-cut)).^2 ) ) , sqrt(mean( (mu_djgp(1+cut:end-cut) - yts(1+cut:end-cut)).^2 ) )]


%figure,
%hold on
%plot(s2_gp,'DisplayName','GP')
%plot(s2_djgp,'m','DisplayName','dJGP')
%title('errors')
%legend('show')

%figure,
%plot(cross_jgp.alpha)

