function C = ftn_h(Y)
% lognormal link function h
% Y: size 2 x J
     %C=log(Y(1,:));
    Y_1 = Y(1, :);
    Y_2 = Y(2, :);
    %Y_2=max(Y_2,0.001);
    C = [1 ./ (2*Y_2); Y_1 ./ Y_2];
end