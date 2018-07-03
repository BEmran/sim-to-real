clear 
clc
syms Jp Ja Mp l r p t phi the dphi dthe g T 

A = [Ja + Jp * sin(the)^2 ,    Mp * l * r * cos(the);...
     Mp * l * r * cos(the),    Jp];
 
B = [1/2 * Jp * dphi^2 * sin(2 * the) + Mp * l * g * sin(the);...
     -Jp * dthe * dphi * sin(2 * the) - Mp * l * r * dthe^2 * sin(the) + T ];

eq = A * B;
eq = simplify(eq,'Steps',100)

d = det([Jp                   ,    -Mp * l * r * cos(the);...
        -Mp * l * r * cos(the),    Ja + Jp * sin(the)^2]);
    

%% Parameters
% Gravity Constant
g = 9.81;
% Motor:-------------------------------------------------------------------
% Resistance
Rm = 8.4;
% Current-torque (N-m/A)
kt = 0.042;
% Back-emf constant (V-s/rad)
km = 0.042;
% Rotary Arm:--------------------------------------------------------------
% Mass (kg)
Mr = 0.095;
% Total length (m)
r = 0.085;
% Moment of inertia about pivot (kg-m^2)
Jr = Mr*r^2/12;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
Dr = 0.0015;
% Pendulum Link:-----------------------------------------------------------
% Mass (kg)
Mp = 0.024;
% Total length (m)
l = 0.129;
% Moment of inertia about pivot (kg-m^2)
Jp = Mp*l^2/12;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
Dp = 0.0005;
