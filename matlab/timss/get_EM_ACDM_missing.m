function [nu, beta, gamma, eta, loglik, phi] = get_EM_ACDM_missing(X, Q, A_in, nu_in, beta_in, gamma_in)

% T, A : functions
% eta_0 : J x 2 matrix 
% EM algorithm given Q

[J, K] = size(Q);
N = size(X, 1);
n_in = size(A_in, 1); % 2^K
index = 1 - isnan(X);

nu = nu_in;
beta = beta_in; % J x (K+1)
gamma = gamma_in;
S = sum(Q, 2);

index_1 = cell(J,1);
for j = 1:J
    index_1{j} = [1, 1+find(Q(j,:))];
end

err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);

% optimization setting for Beta (nonlinear constraints)
options = optimset('Display', 'off'); 

while iter_indicator
    old_loglik = loglik;

    %% E-step
    phi = zeros(N, n_in);
    psi = zeros(n_in, 1);
    exponent=zeros(n_in,1);
    for i = 1:N
        index_j = find(index(i,:));
        for a = 1:n_in
            eta = ftn_h([[1, binary(a-1, K)]*beta'; gamma'])';
            tmp_2 = ftn_A(eta);
            exponent(a) = sum( diag( eta(index_j,:) * ftn_T(X(i,index_j))) - tmp_2(index_j)); 
        end
        logphi_i = exponent + log(nu);
        log_max = max(logphi_i);
        exp_shifted = exp(logphi_i - log_max);
        phi(i,:) = exp_shifted / sum(exp_shifted);
    end
    %% M-step
    % updating nu
    for a = 1:n_in
        psi(a) = sum(phi(:,a));
    end
    nu = psi / sum(psi);
    
    % updating item parameters
    % constraints may vary per each exponential family
    % x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
    for j = 1:J
        index_i = find(index(:,j));
        f = objective_ACDM_missing(phi, X, j, Q(j,:), K, index_i);
        opt = fmincon(f, [nonzeros(beta(j,:)); gamma(j)]', [], [], [], [], ...
            [-2, (-3)*ones(1, S(j)),0], [4*ones(1, S(j) + 1),2], [], options); % need gamma > 0, beta_k > 0 (to avoid sign flip)
        gamma(j) = opt(end);
        beta(j,index_1{j}) = opt(1:end-1);
    end
    
    % update log-lik (up to a constant if there is a h(y) term in the exp fam lik)
    tmp = 0;
    exponent = zeros(n_in, 1);
    for i = 1:N
        index_j = find(index(i,:));
        for a = 1:n_in
            eta = ftn_h([[1, binary(a-1, K)]*beta'; gamma'])';
            tmp_2 = ftn_A(eta);
            exponent(a) = sum( diag( eta(index_j,:) * ftn_T(X(i,index_j))) - tmp_2(index_j));
        end
        tmp = tmp + log(nu' * exp(exponent));
    end
    loglik = tmp;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 9* 1e-2 && itera<1000);
    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

end