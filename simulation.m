function [t,ssim] = simulation(S0,nsim,nsteps,sigma,mu,T)
dt = T/nsteps;
t = [0:dt:T]; %timeline
p = 0.5*( 1 + (mu-0.5*sigma^2)/sigma*sqrt(dt)); %risk-neutral probability in CRR model

ssim = NaN(nsim,nsteps+1);
ssim(:,1) = S0; % start price is S0


for i= 1:nsteps
    updown = 2*((rand(nsim,1)< p)-0.5);
    ssim(:,i+1) = ssim(:,i) .* exp(sigma*sqrt(dt)*updown);
    
end 

end 
    
    




    