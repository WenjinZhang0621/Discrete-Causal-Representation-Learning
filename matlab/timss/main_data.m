%% Main code 

Q = readmatrix('Q.csv', 'NumHeaderLines', 0);%importdata('Q.csv');
Y = importdata('time.csv');

tmp = sum(1-isnan(Y),2);
na.ind = find(tmp ~= 0);
Y = Y(na.ind, :);

N = size(Y,1);
J = size(Y,2);
K = size(Q,2);

A = binary(0:(2^K-1), K);



% fit lognormal-ACDM
C = 20;
nu_est = cell(C,1);
beta_est = cell(C,1);
gamma_est = cell(C,1);
eta_est = cell(C,1);
phi_est = cell(C,1);
loglik = zeros(C,1);

for cc = 1:C
    rng(cc);
    nu_in = drchrnd(5*ones(2^K,1)',1)';
    % if(rand(1) > 0.5) 
    %     nu_in = nu_est{1} ;
    % else
    %     nu_in = drchrnd(5*ones(2^K,1)',1)';
    % end

    g = 2 * ones(J,1) + 0.5*rand(J,1);
    c = 4 * ones(J,1) + 0.5*rand(J,1);

    delta_in = zeros(J, K + 1);
    delta_part_in = zeros(J, K);
    for j = 1:J
        delta_in(j, 1) = g(j);
        delta_part_in(j, Q(j,:) == 1) = drchrnd(5*ones(2,1)',1)'*(c(j)-g(j));
    end
    delta_in(:, 2:end) = delta_part_in;
    sigma_in = 0.5*ones(J,1);

    [nu_est{cc}, beta_est{cc}, gamma_est{cc}, eta_est{cc}, loglik(cc), phi_est{cc}] = ...
        get_EM_ACDM_missing(Y, Q, A, nu_in, delta_in, sigma_in);
end

[~, cc] = max(loglik);


