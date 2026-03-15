%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% TWO-SPECIES COMPETITION %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Master Thesis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Arthur F. Rossignol %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%cd "/working_directory"

clearvars;
close all;

%% PARAMETERS

% environment
l    = 30;      % maximum depth of the water column [m]
l_t  = 10;      % depth of the thermocline [m]
D_e  = 100;     % eddy diffusion coefficient of epilimnion [m²·day⁻¹]
D_h  = 1;       % eddy diffusion coefficient of hypolimnion [m²·day⁻¹]
w_t  = 2;       % width of the thermocline [m]
a_bg = 0.2;     % background turbidity [m⁻¹]

% resources
E   = 0.05;    % sediment interface permeability [m⁻¹]
N_0 = 50;      % sediment nutrient concentration [μg(P)·L⁻¹]
I_0 = 1000;    % light intensity at the surface [μmol(photons)·m⁻²·s⁻¹]
r   = 0.9;     % nutrient recycling rate

% species 1 (BR)
mu_1 = 0.5;     % maximuum growth rate [day⁻¹]
K_1  = 0.2;     % half-saturation constant for nutrient dependency [μg(P)·L⁻¹]
H_1  = 100;     % half-saturation constant for light dependency [μmol(photons)·m⁻²·s⁻¹]
m_1  = 0.2;     % mortality [day⁻¹]
v_1  = 0.3;     % maximum vertical velocity [m·day⁻¹]
q_1  = 1e-3;    % algal nutrient quota [μg(P)·L⁻¹·[cells·mL⁻¹]⁻¹]
a_1  = 1e-5;    % algal absorption coefficient [m⁻¹·[cells·mL⁻¹]⁻¹]
A1_0 = 1000;    % initial algal biomass [cells·mL⁻¹]
    
% species 2 (sinking)
mu_2 = 0.5;     % maximuum growth rate [day⁻¹]
K_2  = 2;       % half-saturation constant for nutrient dependency [μg(P)·L⁻¹]
H_2  = 10;      % half-saturation constant for light dependency [μmol(photons)·m⁻²·s⁻¹]
m_2  = 0.2;     % mortality [day⁻¹]
v_2  = 0.3;     % maximum vertical velocity [m·day⁻¹]
q_2  = 1e-3;    % algal nutrient quota [μg(P)·L⁻¹·[cells·mL⁻¹]⁻¹]
a_2  = 1e-5;    % algal absorption coefficient [m⁻¹·[cells·mL⁻¹]⁻¹]
A2_0 = 1000;    % initial algal biomass [cells·mL⁻¹]

% spatial discretization
dz   = 0.1;
n    = floor(l / dz);
n_t  = floor(l_t / dz);
Z    = linspace(0, l, n);

% vector of parameters
p = [dz; n; n_t; ...
     D_e; D_h; a_bg; r; E; w_t; N_; I_0; ...
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
    I(i) = I(i - 1) - (a_1 * A1(i - 1) + a_2 * A2(i - 1) + a_bg) * I(i - 1) * dz;
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

%% PLOTTING

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

saveas(fig, 'two-species_competition_equilibrium.fig');

% FUNCTION COMPUTING TIME DERIVATIVES

function dU_dt = equations(t, U, p)
    
    % parameter unpacking
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

    % preallocation of vectors
    I        = zeros(n, 1);
    dA1_dt   = zeros(n, 1);
    dV1A1_dz = zeros(n, 1);
    dDA1_dz2 = zeros(n, 1);
    dG1_dz   = zeros(n, 1);
    dA2_dz   = zeros(n, 1);
    dA2_dt   = zeros(n, 1);
    dDA2_dz2 = zeros(n, 1);
    dN_dt    = zeros(n, 1);
    dDN_dz2  = zeros(n, 1);

    % starting values
    A1 = U(1:n);
    A2 = U((n + 1):(2 * n));
    N  = U((2 * n + 1):(3 * n));

    % diffusion
    D  = D_h + (D_e - D_h) ./ (1 + exp(((1:n)' - n_t) * dz / w_t));
    Df = 0.5 * (D(1:(n - 1)) + D(2:n));

    % light intensity
    I(1) = I_0 * exp(- a_0 * 0.5 * dz ...
                     - a_1 * ((3 * A1(1) - A1(2)) / 8 + 3 * A1(1) / 4) * dz ...
                     - a_2 * ((3 * A2(1) - A2(2)) / 8 + 3 * A2(1) / 4) * dz);
    for i = 2:n
        M = - a_1 * ((3 * A1(1) - A1(2)) / 8 + 3 * A1(1) / 4) * dz ...
            - a_2 * ((3 * A2(1) - A2(2)) / 8 + 3 * A2(1) / 4) * dz;
        for k = 2:(i - 1)
            M = M - a_1 * A1(k) * dz ...
                  - a_2 * A2(k) * dz;
        end
        I(i) = I_0 * exp(M - a_0 * (i - 0.5) * dz ...
                           - a_1 * A1(i) * 0.5 * dz ...
                           - a_2 * A2(i) * 0.5 * dz);
    end

    % growth rates
    G1 = mu_1 * min(N ./ (K_1 + N), I ./ (H_1 + I));
    G2 = mu_2 * min(N ./ (K_2 + N), I ./ (H_2 + I));
  
    % fitness gradient of A_BR 
    dG1_dz(2:(n - 1)) = (G1(3:n) - G1(1:(n - 2))) / (2 * dz);
    dG1_dz(1)         = (G1(2) - G1(1)) / dz;
    dG1_dz(n)         = (G1(n) - G1(n - 1)) / dz;

    % vertical velocity of A_BR
    V1  = v_1 * dG1_dz ./ (abs(dG1_dz) + 1e-3);
    V1f = 0.5 * (V1(1:(n - 1)) + V1(2:n));

    % upwind second-order scheme for advection term of A_BR's biomass 
    ii = 3:(n - 2);
    V1f_r = V1f(ii); 
    V1f_l = V1f(ii - 1); 
    tr = (V1f_r > 0);
    tl = (V1f_l > 0);
    dV1A1_dz(ii) = (V1f_r .* (2 * A1(ii) + 5 * A1(ii + 1) - A1(ii + 2)) .* (1 - tr) ...
                 + V1f_r .* (- A1(ii - 1) + 5 * A1(ii) + 2 * A1(ii + 1)) .* tr ...
                 - V1f_l .* ( 2 * A1(ii - 1) + 5 * A1(ii) - A1(ii + 1)) .* (1 - tl) ...
                 - V1f_l .* (- A1(ii - 2) + 5 * A1(ii - 1) + 2 * A1(ii)) .* tl) / (6 * dz);
    t1 = (V1f(1) > 0);
    dV1A1_dz(1) = V1f(1) * t1 * (A1(1) + A1(2)) / (2 * dz) ...
                + V1f(1) * (1 - t1) * (2 * A1(1) + 5 * A1(2) - A1(3)) / (6 * dz);
    t1 = (V1f(2) > 0);
    t2 = (V1f(1) > 0);
    dV1A1_dz(2) = ((- V1f(2) * t1 - 3 * V1f(1) * t2 - 2 * V1f(1) * (1 - t2)) * A1(1) ...
                + ( 2 * V1f(2) * (1 - t1) + 5 * V1f(2) * t1 ...
                - 5 * V1f(1) * (1 - t2) - 3 * V1f(1) * t2) * A1(2) ...
                + ( 5 * V1f(2) * (1 - t1) + 2 * V1f(2) * t1 + V1f(1) * (1 - t2)) * A1(3) ...
                - V1f(2) * (1 - t1) * A1(4)) / (6 * dz);
    t1 = (V1f(n - 1) > 0);
    t2 = (V1f(n - 2) > 0);
    dV1A1_dz(n - 1) = (V1f(n - 2) * t2 * A1(n - 3) ...
                    + (- V1f(n - 1) * t1 - 2 * V1f(n - 2) * (1 - t2) ...
                    - 5 * V1f(n - 2) * t2) * A1(n - 2) ...
                    + ( 5 * V1f(n - 1) * t1 + 3 * V1f(n - 1) * (1 - t1) ...
                    - 5 * V1f(n - 2) * (1 - t2) - 2 * V1f(n - 2) * t2) * A1(n - 1) ...
                    + ( 2 * V1f(n - 1) * t1 + V1f(n - 2) * (1 - t2)) * A1(n)) / (6 * dz);
    t1 = (V1(n) > 0);
    t2 = (V1f(n - 1) > 0);
    dV1A1_dz(n) = V1f(n) * t1 * (- A1(n - 1) + 7 * A1(n)) / (6 * dz) ...
                + V1f(n) * (1 - t1) * A1(n) / dz ...
                - V1f(n - 1) * (1 - t2) * (A1(n) + A1(n - 1)) / (2 * dz) ...
                - V1f(n - 1) * t2 * (2 * A1(n) + 5 * A1(n - 1) - A1(n - 2)) / (6 * dz);

    % upwind second-order scheme for advection term of A_S's biomass 
    dA2_dz(3:(n - 1)) = (2 * A2(4:n) ...
                      + 3 * A2(3:(n - 1)) ...
                      - 6 * A2(2:(n - 2)) ...
                      + A2(1:(n - 3))) / (6 * dz); 
    dA2_dz(1)         = (A2(1) + A2(2)) / (2 * dz);
    dA2_dz(2)         = (- A2(1) + 5 * A2(2) + 2 * A2(3)) / (6 * dz) ...
                      - (A2(1) + A2(2)) / (2 * dz);
    dA2_dz(n)         = (A2(n - 2) - 6 * A2(n - 1) + 5 * A2(n)) / (6 * dz);
    
    % symmetric 2nd-order scheme for diffusion term of A_BR's biomass
    dDA1_dz2(2:(n - 1)) = (Df(2:(n - 1)) .* (A1(3:n)   - A1(2:(n - 1))) ...
                        - Df(1:(n - 2)) .* (A1(2:(n - 1)) - A1(1:(n - 2))) ) / dz^2;
    dDA1_dz2(1)         = Df(1) * (A1(2) - A1(1)) / dz^2;
    dDA1_dz2(n)         = - Df(n - 1) * (A1(n) - A1(n - 1)) / dz^2;
    
    % symmetric 2nd-order scheme for diffusion term of A_S's biomass
    dDA2_dz2(2:(n - 1)) = (Df(2:(n - 1)) .* (A2(3:n) - A2(2:(n - 1))) ...
                        - Df(1:(n - 2)) .* (A2(2:(n - 1)) - A2(1:(n - 2)))) / dz^2;
    dDA2_dz2(1)         = Df(1) * (A2(2) - A2(1)) / dz^2;
    dDA2_dz2(n)         = - Df(n - 1) * (A2(n) - A2(n - 1)) / dz^2;

    % symmetric 2nd-order scheme for diffusion term of nutrients
    dDN_dz2(2:(n - 1)) = (Df(2:(n - 1)) .* (N(3:n) - N(2:(n - 1))) ...
                       - Df(1:(n - 2)) .* (N(2:(n - 1)) - N(1:(n - 2))) ) / dz^2;
    dDN_dz2(1)         = Df(1) * (N(2) - N(1)) / dz^2;
    dDN_dz2(n)         = D(n) * E * (N_0 - N(n)) / dz ...
                       - Df(n - 1) * (N(n) - N(n - 1)) / dz^2;
                       
    % time derivatives of A_BR's biomass, A_S's biomass, nutrients
    dA1_dt(1:n) = (G1(1:n) - m_1) .* A1(1:n) - dV1A1_dz(1:n) + dDA1_dz2(1:n);
    dA2_dt(1:n) = (G2(1:n) - m_2) .* A2(1:n) - v_2 * dA2_dz(1:n) + dDA2_dz2(1:n);
    dN_dt(1:n)  = q_1 * (r * m_1 - G1(1:n)) .* A1(1:n) ...
                + q_2 * (r * m_2 - G2(1:n)) .* A2(1:n) ...
                + dDN_dz2(1:n);
    
    % output
    dU_dt = [dA1_dt; dA2_dt; dN_dt];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
