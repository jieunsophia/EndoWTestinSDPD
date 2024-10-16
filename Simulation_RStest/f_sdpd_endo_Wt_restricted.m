function llike = f_sdpd_endo_Wt_restricted(parm,y,wy,R1,kx1,z,R2,W)

% Under the joint null: lambda0=gammma0=rho0=0
R1=R1(:,3:(3+kx1-1));

[junk,kR2]=size(R2);
[junk,kz]=size(z);
[kparm,junk]=size(parm);

phi2=parm(1:(kz*kR2));
Phi2=reshape(phi2,kR2,kz);

vSigma=parm((1+kz*kR2):kparm);
Sigma=zeros(kz,kz);
for i=1:kz
    Sigma(i:kz,i)=vSigma(((i-1)*kz+1-(i-1)*(i-2)/2):(i*kz-i*(i-1)/2));
end
Sigma=Sigma+Sigma';
for i=1:kz
    Sigma(i,i)=0.5*Sigma(i,i);
    Sigma(i,i)=exp(Sigma(i,i));
end

[n junk]=size(W);
[nt junk]=size(y);
[junk kR1]=size(R1);
t=nt/n; 

J=0;
for i=1:t
    St=speye(n);
    dSt=log(det(St));
    J=J+dSt;
end

Sy=y;

epsilon=z-R2*Phi2;
x=R1;
b = x\Sy;
e=Sy-x*b; 
epe = e'*e;

Sigmai=inv(Sigma);
temp=0;
for i=1:t
    for j=1:kz
        for k=1:kz
            temp=temp+Sigmai(j,k)*epsilon((1+(i-1)*n):(i*n),j)'*epsilon((1+(i-1)*n):(i*n),k);
        end
    end
end
llike = (n/2)*log(epe/nt) + (n/2)*log(det(Sigma)) - J/t + (0.5/t)*temp + (n/2)*(log(2*pi)+1);

