
%%%%%%%%%%%%%%%%%%%%%%%%% TWO-SPECIES COMPETITION %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Arthur F. Rossignol %%%%%%%%%%%%%%%%%%%%%%%%%%%

cd "/working_directory"

clearvars;
close all;

%% PARAMETERS

% environment
l   = 30;       % maximum depth of the water column 
l_t = 10;       % depth of the thermocline
D_e = 100;      % eddy diffusion coefficient of epilimnion
D_h = 1;        % eddy diffusion coefficient of hypolimnion
a_0 = 0.2;      % background turbidity
r   = 0.9;      % nutrient recycling rate
E   = 0.05;     % sediment layer permeability
w_t = 2;        % width of the thermocline 
N_0 = 50;       % sediment nutrient concentration
I_0 = 1000;     % light intensity at the surface

% species 1
mu_1 = 0.5;     % maximuum growth rate
K_1  = 0.2;     % half-saturation constant for nutrient dependency
H_1  = 100;     % half-saturation constant for light dependency
m_1  = 0.2;     % mortality
v_1  = 0.3;     % maximum vertical velocity
q_1  = 1e-3;    % algal nutrient quota
a_1  = 1e-5;    % algal absorption coefficient  
A1_0 = 1000;    % initial algal biomass
    
% species 2
mu_2 = 0.5;     % maximuum growth rate
K_2  = 2;       % half-saturation constant for nutrient dependency
H_2  = 10;      % half-saturation constant for light dependency
m_2  = 0.2;     % mortality
v_2  = 0.3;     % maximum vertical velocity
q_2  = 1e-3;    % algal nutrient quota
a_2  = 1e-5;    % algal absorption coefficient  
A2_0 = 1000;    % initial algal biomass

% spatial discretization
dz   = 0.15;
n    = floor(l / dz);
n_t  = floor(l_t / dz);
Z    = linspace(0, l, n);

% vector of parameters
p = [dz; n; n_t; ...
     D_e; D_h; a_0; r; E; w_t; N_; I_0; ...
     mu_1; K_1; H_1; m_1; v_1; q_1; a_1; ...
     mu_2; K_2; H_2; m_2; v_2; q_2; a_2];

% time
t_max = 1e5;

% solver options
options = odeset('nonnegative', 1, 'RelTol', 1e-4, 'AbsTol', 1e-8);

%% SOLVING

% initial profiles
U0 = [A1_0 * ones(n, 1);
      A2_0 * ones(n, 1);
      ones(n_t, 1); transpose(linspace(1, N_0, n - n_t))];

% solver calling
[t, sol] = ode15s(@(t, U) equations(t, U, p), [0, t_max], U0, options);

% equilibrium solutions
A1 = sol(end, 1:n);
A2 = sol(end, (n + 1):(2 * n));

%% POST-PROCESSING

% computation of total biomass
A1_e = 0;   % total biomass of S1 in epilimnion
A1_h = 0;   % total biomass of S1 in hypolimnion
A2_e = 0;   % total biomass of S2 in epilimnion
A2_h = 0;   % total biomass of S2 in hypolimnion
for i = 1:floor(n_t)
    A1_u = A1_u + A1(i) * dz;
    A2_u = A2_u + A2(i) * dz;
end
for i = floor(n_t + 1):n
    A1_d = A1_d + A1(i) * dz;
    A2_d = A2_d + A2(i) * dz;
end
A1_tot = A1_u + A1_d;   % total biomass of S1 in the water column
A2_tot = A2_u + A2_d;   % total biomass of S2 in the water column

% vectors
I      = zeros(n, 1);
D      = zeros(n, 1);
G1     = zeros(n, 1);
V1     = zeros(n, 1);
dG1_dz = zeros(n, 1);
G2     = zeros(n, 1);
dG2_dz = zeros(n, 1);
V2     = zeros(n, 1);

% computation of light intensity
I(1) = I_0;
for i = 2:n
    I(i) = I(i - 1) - (a_1 * A1(i - 1) + a_2 * A2(i - 1) + a_0) * I(i - 1) * dz;
end
    
% computation of growth rates
for i = 1:n
    G1(i) = mu_1 * min((N(i) / (K_1 + N(i))), (I(i) / (H_1 + I(i)))) - m_1;
    G2(i) = mu_2 * min((N(i) / (K_2 + N(i))), (I(i) / (H_2 + I(i)))) - m_2;
end

% computation of fitness gradients
for i = 2:(n - 1)
    dG1_dz(i) = (G1(i + 1) - G1(i - 1)) / (2 * dz);
    dG2_dz(i) = (G2(i + 1) - G2(i - 1)) / (2 * dz);
end
dG1_dz(1) = (G1(2) - G1(1)) / dz;
dG1_dz(n) = (G1(n) - G1(n - 1)) / dz;
dG2_dz(1) = (G2(2) - G2(1)) / dz;
dG2_dz(n) = (G2(n) - G2(n - 1)) / dz;

% computation of vertical velocities & eddy diffusion
for i = 1:n
    V1(i) = v_1 * dG1_dz(i) / (abs(dG1_dz(i)) + 1e-6);
    V2(i) = v_2;
    D(i) = D_e + (D_h - D_e) / (1 + exp((i - n_t) / w_t));
end

%% PLOTING

fig = figure(1);

subplot(3, 4, 1);
plot(A1, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('algal biomass of S1');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 2);
plot(G1, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('net growth rate of S1');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 3);
plot(dG1_dz, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('fitness gradient of S1');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 4);
plot(V1, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('vertical velocity of S1 (BR)');
xlabel('depth');
set(gca, 'YDir', 'reverse');

subplot(3, 4, 5);
plot(A2, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('algal biomass of S2');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 6);
plot(G2, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('net growth rate of S2');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 7);
plot(dG2_dz, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('fitness gradient of alga 2');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 8);
plot(V2, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('vertical velocity of S2 (sinking)');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 9);
plot(A1 + A2, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('total algal biomass');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 10);
plot(N, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('nutreint concentration');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 11);
plot(I, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('light intensity');
xlabel('depth');
set(gca, 'YDir','reverse');

subplot(3, 4, 12);
plot(D, linspace(0, n * dz, n), 'LineWidth', 2.5);
title('eddy diffusion coefficient');
xlabel('depth');
set(gca, 'YDir','reverse');

saveas(fig, 'equilibrium.fig');

%% FUNCTION COMPUTING TIME DERIVATIVES

function dU_dt = equations(t, U, p)

    % parameters
    dz   = p(1);
    n    = p(2);
    n_t  = p(3);
    D_e  = p(4);
    D_h  = p(5);
    a_0  = p(6);
    r    = p(7);
    E    = p(8);
    w_t  = p(9);
    N_0  = p(10);
    I_0  = p(11);
    mu_1 = p(12);
    K_1  = p(13);
    H_1  = p(14);
    m_1  = p(15);
    v_1  = p(16);
    q_1  = p(17);
    a_1  = p(18);
    mu_2 = p(19);
    K_2  = p(20);
    H_2  = p(21);
    m_2  = p(22);
    v_2  = p(23);
    q_2  = p(24);
    a_2  = p(25);

    % initialization of vectors
    I        = zeros(n, 1);
    D        = zeros(n, 1);
    dA1_dz   = zeros(n, 1);
    dA1_dz2  = zeros(n, 1);
    dA1_dt   = zeros(n, 1);
    dV1A1_dz = zeros(n, 1);
    dDA1_dz2 = zeros(n, 1);
    G1       = zeros(n, 1);
    dG1_dz   = zeros(n, 1);
    V1       = zeros(n, 1);
    dA2_dz   = zeros(n, 1);
    dA2_dz2  = zeros(n, 1);
    dA2_dt   = zeros(n, 1);
    dV2A2_dz = zeros(n, 1);
    dDA2_dz2 = zeros(n, 1);
    G2       = zeros(n, 1);
    dG2_dz   = zeros(n, 1);
    V2       = zeros(n, 1);
    dN_dz    = zeros(n, 1);
    dN_dz2   = zeros(n, 1);
    dN_dt    = zeros(n, 1);
    dDN_dz2  = zeros(n, 1);

    % starting values of A1, A2, N
    A1 = U(1:n);
    A2 = U((n + 1):(2 * n));
    N  = U((2 * n + 1):(3 * n));

    % spatial derivatives of A1
    dA1_dz(2:(n - 1))  = (A1(3:n) - A1(1:(n - 2))) / (2 * dz);
    dA1_dz2(2:(n - 1)) = (A1(3:n) - 2 * A1(2:(n - 1)) + A1(1:(n - 2))) / (dz^2);

    % spatial derivatives of A2
    dA2_dz(2:(n - 1))  = (A2(3:n) - A2(1:(n - 2))) / (2 * dz);
    dA2_dz2(2:(n - 1)) = (A2(3:n) - 2 * A2(2:(n - 1)) + A2(1:(n - 2))) / (dz^2);

    % spatial derivatives of N
    dN_dz(2:(n - 1))  = (N(3:n) - N(1:(n - 2))) / (2 * dz);
    dN_dz2(2:(n - 1)) = (N(3:n) - 2 * N(2:(n - 1)) + N(1:(n - 2))) / (dz^2);

    % computation of eddy diffusion (D)
    for i = 1:n
        D(i) = D_e + (D_h - D_e) / (1 + exp((i - n_t) / w_t));
    end

    % computation of light intensity (I)
    I(1) = I_0;
    for i = 2:n
        I(i) = I(i - 1) - (a_1 * A1(i - 1) + a_2 * A2(i - 1) + a_0) * I(i - 1) * dz;
    end
    
    % computation of growth rates (G1 & G2)
    for i = 1:n
        G1(i) = mu_1 * min((N(i) / (K_1 + N(i))), (I(i) / (H_1 + I(i)))) - m_1;
        G2(i) = mu_2 * min((N(i) / (K_2 + N(i))), (I(i) / (H_2 + I(i)))) - m_2;
    end

    % computation of fitness gradients (dG1_dz & dG2_dz)
    for i = 2:(n - 1)
        dG1_dz(i) = (G1(i + 1) - G1(i - 1)) / (2 * dz);
        dG2_dz(i) = (G2(i + 1) - G2(i - 1)) / (2 * dz);
    end
    dG1_dz(1) = (G1(2) - G1(1)) / dz;
    dG1_dz(n) = (G1(n) - G1(n - 1)) / dz;
    dG2_dz(1) = (G2(2) - G2(1)) / dz;
    dG2_dz(n) = (G2(n) - G2(n - 1)) / dz;
    
    % computation of vertical velocities (V1 & V2)
    for i = 1:n
        V1(i) = v_1 * dG1_dz(i) / (abs(dG1_dz(i)) + 1e-6);
        V2(i) = v_2;
    end

    % boundary conditions at the surface
    dA1_dz(1) = (V1(1) / D(1)) * A1(1);
    dA2_dz(1) = (V2(1) / D(1)) * A2(1);
    dN_dz(1)  = 0;
    
    % boundary conditions at the bottom
    dA1_dz(n) = (V1(n) / D(n)) * A1(n);
    dA2_dz(n) = (V2(n) / D(n)) * A2(n);
    dN_dz(n)  = E * (N_0 - N(n));

    % 1st-order spatial derivative of (V1 * A1) 
    dV1A1_dz(3:(n - 2)) = (- V1(5:n) .* A1(5:n) ...
                           + 8 * V1(4:(n - 1)) .* A1(4:(n - 1)) ...
                           - 8 * V1(2:(n - 3)) .* A1(2:(n - 3)) ...
                           + V1(1:(n - 4)) .* A1(1:(n - 4))) / (12 * dz);
    dV1A1_dz(1)     = (V1(2) * A1(2) - V1(1) * A1(1)) / dz;
    dV1A1_dz(2)     = (V1(3) * A1(3) - V1(1) * A1(1)) / (2 * dz);
    dV1A1_dz(n - 1) = (V1(n) * A1(n) - V1(n - 2) * A1(n - 2)) / (2 * dz);
    dV1A1_dz(n)     = (V1(n) * A1(n) - V1(n - 1) * A1(n - 1)) / dz;

    % 1st-order spatial derivative of (V2 * A2)
    dV2A2_dz(3:(n - 2)) = (- V2(5:n) .* A2(5:n) ...
                           + 8 * V2(4:(n - 1)) .* A2(4:(n - 1)) ...
                           - 8 * V2(2:(n - 3)) .* A2(2:(n - 3)) ...
                           + V2(1:(n - 4)) .* A2(1:(n - 4))) / (12 * dz);
    dV2A2_dz(1)     = (V2(2) * A2(2) - V2(1) * A2(1)) / dz;
    dV2A2_dz(2)     = (V2(3) * A2(3) - V2(1) * A2(1)) / (2 * dz);
    dV2A2_dz(n - 1) = (V2(n) * A2(n) - V2(n - 2) * A2(n - 2)) / (2 * dz);
    dV2A2_dz(n)     = (V2(n) * A2(n) - V2(n - 1) * A2(n - 1)) / dz;

    % values of 2nd-order spatial derivatives at boundaries
    dA1_dz2(1) = (dA1_dz(2) - dA1_dz(1)) / dz;
    dA1_dz2(n) = (dA1_dz(n) - dA1_dz(n - 1)) / dz;
    dA2_dz2(1) = (dA2_dz(2) - dA2_dz(1)) / dz;
    dA2_dz2(n) = (dA2_dz(n) - dA2_dz(n - 1)) / dz;
    dN_dz2(1)  = (dN_dz(2) - dN_dz(1)) / dz;
    dN_dz2(n)  = (dN_dz(n) - dN_dz(n - 1)) / dz;
    
    % 1st-order spatial derivative of (D * dA1_dz)
    for i = 2:(n - 1)
        dDA1_dz2(i) = (D(i + 1) - D(i - 1)) / dz * dA1_dz(i) + D(i) * dA1_dz2(i);
    end    
    dDA1_dz2(1) = (D(2) - D(1)) / dz * dA1_dz(1) + D(1) * dA1_dz2(1);
    dDA1_dz2(n) = (D(n) - D(n - 1)) / dz * dA1_dz(n) + D(n) * dA1_dz2(n);

    % 1st-order spatial derivative of (D * dA2_dz)
    for i = 2:(n - 1)
        dDA2_dz2(i) = (D(i + 1) - D(i - 1)) / dz * dA2_dz(i) + D(i) * dA2_dz2(i);
    end    
    dDA2_dz2(1) = (D(2) - D(1)) / dz * dA2_dz(1) + D(1) * dA2_dz2(1);
    dDA2_dz2(n) = (D(n) - D(n - 1)) / dz * dA2_dz(n) + D(n) * dA2_dz2(n);

    % 1st-order spatial derivative of (D * dN_dz)
    for i = 2:(n - 1)
        dDN_dz2(i) = (D(i + 1) - D(i - 1)) / dz * dN_dz(i) + D(i) * dN_dz2(i);
    end
    dDN_dz2(1) = (D(2) - D(1)) / dz * dN_dz(1) + D(1) * dN_dz2(1);
    dDN_dz2(n) = (D(n) - D(n - 1)) / dz * dN_dz(n) + D(n) * dN_dz2(n);

    % time derivatives of A1, A2, N
    for i = 1:n 
        dA1_dt(i) = G1(i) * A1(i) - dV1A1_dz(i) + dDA1_dz2(i);
        dA2_dt(i) = G2(i) * A2(i) - dV2A2_dz(i) + dDA2_dz2(i);
        dN_dt(i)  = q_1 * (r * m_1 - mu_1 * g(N(i), I(i), K_1, H_1)) * A1(i) ...
                     + q_2 * (r * m_2 - mu_2 * g(N(i), I(i), K_2, H_2)) * A2(i) ...
                     + dDN_dz2(i);
    end

    % output
    dU_dt = [dA1_dt; dA2_dt; dN_dt];

end
