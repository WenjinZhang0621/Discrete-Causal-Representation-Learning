function C = ftn_A(eta)
% lognormal normalizing constant A
     %C=exp(eta);
   %eta_1 = max(eta(:,1),0.1);
   eta_1=eta(:,1);
    eta_2 = eta(:,2);
    C = eta_2 .^2 ./ (4 * eta_1) + log( 1 ./ (2 * eta_1)) /2;
end