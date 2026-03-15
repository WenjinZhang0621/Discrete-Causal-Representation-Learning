%% main.m

this_file = mfilename("fullpath");
this_dir  = fileparts(this_file);
repo_root = fileparts(fileparts(this_dir));

q_path   = fullfile(repo_root, "data", "TIMSS", "Q.csv");
y_path   = fullfile(repo_root, "data", "TIMSS", "time.csv");
out_path = fullfile(repo_root, "data", "TIMSS", "timss_matlab_estimates.mat");

Q = readmatrix(q_path, 'NumHeaderLines', 0);
Y = readmatrix(y_path);

tmp = sum(1 - isnan(Y), 2);
na_ind = find(tmp ~= 0);
Y = Y(na_ind, :);

N = size(Y,1);
J = size(Y,2);
K = size(Q,2);

A = binary(0:(2^K-1), K);

% fit lognormal-ACDM
C = 20;
nu_est = cell(C,1);
beta_cell = cell(C,1);
gamma_cell = cell(C,1);
eta_est = cell(C,1);
phi_est = cell(C,1);
loglik = zeros(C,1);

for cc = 1:C
    rng(cc);
    nu_in = drchrnd(5*ones(2^K,1)',1)';

    g = 2 * ones(J,1) + 0.5*rand(J,1);
    c = 4 * ones(J,1) + 0.5*rand(J,1);

    delta_in = zeros(J, K + 1);
    delta_part_in = zeros(J, K);
    for j = 1:J
        delta_in(j, 1) = g(j);
        delta_part_in(j, Q(j,:) == 1) = drchrnd(5*ones(2,1)',1)' * (c(j)-g(j));
    end
    delta_in(:, 2:end) = delta_part_in;
    sigma_in = 0.5*ones(J,1);

    [nu_est{cc}, beta_cell{cc}, gamma_cell{cc}, eta_est{cc}, loglik(cc), phi_est{cc}] = ...
        get_EM_ACDM_missing(Y, Q, A, nu_in, delta_in, sigma_in);
end

[~, best_cc] = max(loglik);

p_est = nu_est{best_cc};
beta_est = beta_cell{best_cc};
gamma_est = gamma_cell{best_cc};

save(out_path, "p_est", "beta_est", "gamma_est", "best_cc", "loglik");
fprintf("Saved estimates to:\n%s\n", out_path);
