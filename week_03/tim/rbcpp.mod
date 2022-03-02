//----------------------------------------------------------------
// RBCPP.mod
//
// DYNARE++ mod file - RBC model 
// (separable log utilities, Cobb-Douglas production)
//
// Modified: T.Kam from original DYNARE example .MOD file
// Changelog: 2022-02-24
//----------------------------------------------------------------

//----------------------------------------------------------------
// 1. Defining variables
//----------------------------------------------------------------

var y c k i l y_l z;
varexo e;

parameters beta psi delta alpha rho;

//----------------------------------------------------------------
// 2. Calibration
//----------------------------------------------------------------

alpha   = 0.33;
beta    = 0.99;
delta   = 0.023;
psi     = 1.75;
rho     = 0.95;
// sigma   = (0.007/(1-alpha)); // place in vcov directly!

//----------------------------------------------------------------
// 3. Model
//----------------------------------------------------------------

model;
  (1/c) = beta*(1/c(+1))*(1+alpha*(k^(alpha-1))*(exp(z(+1)) 
                                         *l(+1))^(1-alpha)-delta);
  psi*c/(1-l) = (1-alpha)*(k(-1)^alpha)*(exp(z)^(1-alpha))
                                                    *(l^(-alpha));
  c+i = y;
  y = (k(-1)^alpha)*(exp(z)*l)^(1-alpha);
  i = k-(1-delta)*k(-1);
  y_l = y/l;
  z = rho*z(-1)+e;
end;

//----------------------------------------------------------------
// 4. Computation
//----------------------------------------------------------------

// Guess of steady-state solution (root-solved)
initval;
  z = 0;
  e = 0;
  k = 2.1;
  c = 0.76;
  l = 0.3;
  y = (k^alpha)*(exp(z)*l)^(1-alpha);
  i = delta*k;
  y_l = y/l; 
end;

// Shock variance (or variance-covariance matrix)
// Here, set: sigma^2
vcov = [ 0.00010915571396747609 ];

// Order of approximation
order = 1;
