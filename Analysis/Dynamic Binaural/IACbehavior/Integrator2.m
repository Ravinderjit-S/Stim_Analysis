function y = Integrator2(t,A,B,C,tau1,tau2)

% A = abs(A); B = abs(B); C = abs(C);
y = A *(1 - B*exp(-t/tau1) - C*exp(-t/tau2));

end


