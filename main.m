load('ass1.mat')
%return take IBM as an example

re = (price(2:end,:)./price(1:end-1,:)) -1 ;
mu = mean(re);
Q = cov(re);

%TAKE ibm as an example

mu = mu(1);
sigma = sqrt(Q(1,1));

S0 = 158.21; % "today's price"
nsims = 10000; 

T = 365; %DAYS 1 year later
nsteps = 3650;
nsim = 10000;

[t , ssim] =  simulation(S0,nsim,nsteps,sigma,mu,T); 

% if no viotility stock price

s = S0*exp(mu*365); %320.01

% see how many simulation is above this price

a = ssim(:,3651)>= s;

number = nnz(a);

percentage = number/nsim; % 46.31% of probability that IBM performs well





