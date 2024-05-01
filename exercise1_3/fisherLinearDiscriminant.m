function v = fisherLinearDiscriminant(X1, X2)
    m1 = size(X1, 1);
    m2 = size(X2, 1);
    m = m1 + m2;
    
    mu1 = mean(X1);
    mu2 = mean(X2);
    
    S1 = 1/m1 * X1.' * X1;
    S2 = 1/m2 * X2.' * X2;

    Sw = m1/m.*S1 + m2/m.*S2;

    v = inv(Sw)*(mu1 - mu2).';
    
    v = v/norm(v);
