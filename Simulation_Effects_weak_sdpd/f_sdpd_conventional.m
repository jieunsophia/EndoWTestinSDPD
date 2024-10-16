function llike = f_sdpd_conventional(parm,y,wy,R1,W)

lambda=2/(1+exp(-parm(1)))-1;

[n junk]=size(W);
[nt junk]=size(y);
t=nt/n; 

J=0;
for i=1:t
    St=speye(n)-lambda*W(:,1+i*n:(i+1)*n);
    dSt=log(det(St));
    J=J+dSt;
end

Sy=y-lambda*wy;
b=R1\Sy;
e=Sy-R1*b;
epe = e'*e;

%llike =(n/2)*log(epe/nt) +(n/2)*log(det(Sigma))-J/t+(0.5/t)*temp;
llike =(n/2)*log(epe/nt) -J/t + (n/2)*(log(2*pi)+1);