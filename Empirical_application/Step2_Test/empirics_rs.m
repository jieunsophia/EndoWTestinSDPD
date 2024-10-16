clear all; clc; warning off; info.lflag=0; 

% = Experimental Input =============================

%W_type="econ";
W_type="composite";

%durbin=0; % SAR
durbin=1; % SDM

% = Data load ============================================================
data='data_analysis.csv'; 
Data=csvread(data,1,1); n=55;
[L ncol]=size(Data); T=L/n;
th=1;

% Year
year=Data(:,th); th=th+1;

% Geo info
xlong=Data(:,th); th=th+1;
ylat=Data(:,th); th=th+1;
geo=[xlong(1:n) ylat(1:n)];

% Outcome: lny
outcome=Data(:,th); th=th+1;
%outcome=(outcome-min(outcome))/(max(outcome)-min(outcome)); % Rescale to [0,1]

% - Covariates & triggers for spillover -----------------------------------
cols_covariates=th:ncol;
cols_triggers=[7];
covariates=Data(:,setdiff(cols_covariates, cols_triggers));
triggers=Data(:,cols_triggers); 

% - Declare variables ------------------------------
y=outcome;
%x=[covariates triggers];
x=covariates; [~, k1]=size(x);
z=triggers; [junk p]=size(z); 
zx=covariates; 
lny_lagged=1;

% = Construct W ==================================================

% - Geographic distance ---------------------------
% radian
X=(pi/180)*geo(:,2);
Y=(pi/180)*geo(:,1);

% Distance arc matrix (D)
T1=repmat(X',n,1);
T2=repmat(Y',n,1);
T3=repmat(X,1,n);
T4=repmat(Y,1,n);
D=(6378 * acos(cos(abs(T2-T4)).*cos(T1).*cos(T3) + sin(T1).*sin(T3))).*(ones(n,n)-eye(n,n));

% Rescale to [0,1]
%D=reshape(D,[],1);
%D=(D-min(D))/(max(D)-min(D)); % Rescale to [0,1]
%D=reshape(D,n,n);

% Inverse distance 
%W_d=1./D;
%W_d(1:n+1:end)=0;
W_d=(ones(n)./(D+eye(n))).^2 - eye(n);
W_d=kron(eye(T),W_d);

% - Socioeconomic distance ---------------------------
E=zeros(L,L);
for t=1:T
    for i=1:n
        for j=(t-1)*n+(i+1):(t-1)*n+n
            E((t-1)*n+i,j)=abs(z((t-1)*n+i)-z(j));
        end
    end
end
E=E+E';

% Rescale to [0,1]
E=reshape(E,[],1);
E=(E-min(E))/(max(E)-min(E));
E=reshape(E,L,L);

% Inverse distance
W_e=zeros(L,L);
for t=1:T
    for i=1:n
        for j=((t-1)*n+(i+1)):((t-1)*n+n)
           W_e((t-1)*n+i,j)=1./E((t-1)*n+i,j);
        end
    end
end
W_e=W_e+W_e';
W_e(isinf(W_e))=max(W_e(~isinf(W_e)))+1; % Replace 'Inf' with the max_value+1

% Spatial weight matrix
if W_type=="econ"
    W=W_e;
elseif W_type=="composite"
    W=W_d.*W_e;
end

% Sparsity
W(W<quantile(W(W~=0),0.8))=0;

% Row-normalize W & Declare W1, W2, and W3
W1=normw(W);
W2=zeros(n*T,n*T); W3=zeros(n*T,n*T);
for t=1:T-1
    W2(n*t+1:n*(t+1),(t-1)*n+1:t*n)=eye(n);
    W3(n*t+1:n*(t+1),(t-1)*n+1:t*n)=W1((t-1)*n+1:t*n,(t-1)*n+1:t*n);
end

% Dimension change for W
w=zeros(n,n*(T+1));
for t=1:T+1
    if t==1
        w(:,1:n)=W1(1:n,1:n); % W at t=0: Replicate (W at t=0) using (W at t=1)
    else
        w(:,(t-1)*n+1:t*n)=W1((t-2)*n+1:(t-1)*n,(t-2)*n+1:(t-1)*n); % W at t>=1
    end
end
W=w;

% Predict unknown initial conditions for y0 and z0
y0_proxy=zeros(n,1); z0_proxy=zeros(n,p);
w1=w(1:n,1:n); y1=y(1:n); z1=z(1:n,:);
for i=1:n
    current_row=w1(i, :); % Extract the current row
    positive_indices=find(current_row>0); % Find indices of positive values
    positive_values=current_row(positive_indices); % Get the positive values
    [~, sorted_indices]=sort(positive_values,'descend'); % Sort the positive values to find the largest ones
    num_to_take=min(10,length(sorted_indices)); % Get the original indices of the top positive values, up to a maximum of 10
    top_positive_indices=positive_indices(sorted_indices(1:num_to_take));
    y0_proxy(i)=mean(y1(top_positive_indices));
    z0_proxy(i,:)=mean(z1(top_positive_indices,:));
end

y=[y0_proxy; y];
z=[z0_proxy; z];

% Add x if durbin==1
if (durbin==1) 
    x=[x W1*x]; [~, k1]=size(x);
end

% = Estimation (SDPD) =====================================================            
info = struct('n',n,'t',t,'rmin',0,'rmax',1,'lflag',0,'tl',1,'stl',1,'tlz',1,'tly',lny_lagged);            
fields = fieldnames(info);
nf = length(fields);
for i=1:nf
    if strcmp(fields{i},'tl')
            tl = info.tl; % star lag index
    elseif strcmp(fields{i},'stl')
            stl = info.stl; % star lag index
    elseif strcmp(fields{i},'tly')
            tly = info.tly; % star lag index
    elseif strcmp(fields{i},'tlz')
            tlz = info.tlz; % star lag index
    end
end

[n junk] = size(W); [L junk]=size(y); [junk kx]=size(x);
[junk kz]=size(z); [junk kzx]=size(zx);
t=L/n-1; L=n*t;

% Remove time effects
y2=y; x2=x; z2=z; zx2=zx;

y2temp=y2; x2temp=x2; z2temp=z2; zx2temp=zx2;
Jn=speye(n)-1/n*ones(n,1)*ones(1,n);
for i=1:t+1
    y2temp(1+(i-1)*n:i*n,:)=Jn*y2(1+(i-1)*n:i*n,:);
    z2temp(1+(i-1)*n:i*n,:)=Jn*z2(1+(i-1)*n:i*n,:);
end
for i=1:t
    x2temp(1+(i-1)*n:i*n,:)=Jn*x2(1+(i-1)*n:i*n,:);
    zx2temp(1+(i-1)*n:i*n,:)=Jn*zx2(1+(i-1)*n:i*n,:);
end
y2=y2temp; x2=x2temp; z2=z2temp; zx2=zx2temp;

yt=y2(n+1:n+L);
ytl=y2(1:L);
ysl=zeros(L,1); ystl=zeros(L,1);
for i=1:t
    ysl(1+(i-1)*n:i*n)=W(:,1+i*n:(i+1)*n)*yt(1+(i-1)*n:i*n);
    ystl(1+(i-1)*n:i*n)=W(:,1+(i-1)*n:i*n)*ytl(1+(i-1)*n:i*n);
end

% Remove individual effects
yt=reshape(yt,n,t);
temp=mean(yt')';
temp=temp*ones(1,t);
yt=yt-temp;
yt=reshape(yt,L,1);

ytl=reshape(ytl,n,t);
temp=mean(ytl')';
temp=temp*ones(1,t);
ytl=ytl-temp;
ytl=reshape(ytl,L,1);

ysl=reshape(ysl,n,t);
temp=mean(ysl')';
temp=temp*ones(1,t);
ysl=ysl-temp;
ysl=reshape(ysl,L,1);

ystl=reshape(ystl,n,t);
temp=mean(ystl')';
temp=temp*ones(1,t);
ystl=ystl-temp;
ystl=reshape(ystl,L,1);

zt=z2(n+1:n+L,:);
ztl=z2(1:L,:);

zt=reshape(zt,n,t,kz);
for i=1:kz
    temp=mean(zt(:,:,i)')';
    temp=temp*ones(1,t);
    zt(:,:,i)=zt(:,:,i)-temp;
end
zt=reshape(zt,L,kz);

ztl=reshape(ztl,n,t,kz);
for i=1:kz
    temp=mean(ztl(:,:,i)')';
    temp=temp*ones(1,t);
    ztl(:,:,i)=ztl(:,:,i)-temp;
end
ztl=reshape(ztl,L,kz);

xt=x2;
if isempty(x) == 0
    [junk,kx]=size(xt);
    xt=reshape(xt,n,t,kx);
    for i=1:kx
        temp=mean(xt(:,:,i)')';
        temp=temp*ones(1,t);
        xt(:,:,i)=xt(:,:,i)-temp;
    end
    xt=reshape(xt,L,kx);
else
    xt=[];
end

zxt=zx2;
if isempty(zx) == 0
    [junk,kzx]=size(zxt);
    zxt=reshape(zxt,n,t,kzx);
    for i=1:kzx
        temp=mean(zxt(:,:,i)')';
        temp=temp*ones(1,t);
        zxt(:,:,i)=zxt(:,:,i)-temp;
    end
    zxt=reshape(zxt,L,kzx);
else
    zxt=[];
end

% Regressors in the main equation
if stl + tl == 2
    R1t=[ytl ystl xt];
elseif stl + tl == 1
    if stl == 1, R1t=[ystl xt]; else R1t=[ytl xt]; end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end

% Regressors in the auxiliary equation
if tly + tlz == 2
    R2t=[ztl ytl zxt];
elseif tly + tlz == 1
    if tly == 1, R2t=[ytl zxt]; else R2t=[ztl zxt]; end
elseif tly + tlz == 0
    R2t=zxt;
else
    error('Double-Check tly & tlz # in Info structure ');
end

[junk kR1]=size(R1t);
[junk kR2]=size(R2t);


% Restricted ML estimators (null: lambda0=gamma0=rho0=0)
options = optimset('Display','iter-detailed','PlotFcns', @optimplotfval,'TolX', 1e-4, ...
                        'MaxTime', 1800, 'MaxIter', 50000, 'MaxFunEvals', 50000);

J=kz*(kz+1)/2; % number of distinct elemeLs in Sigma_epsilon
parm = zeros(kz*(kz*tlz+tly+kzx)+J,1); % ztl + ytl + zxt + Sigma_epsilon
[kparm,junk]=size(parm);

parm_z=R2t\zt; % specify initial value of Z equation
e_z=zt-R2t*parm_z;
cov_z=cov(e_z);
a_z=zeros(J,1);
for i=1:kz
    a_z((i-1)*kz+1-(i-1)*(i-2)/2:i*kz-i*(i-1)/2)=cov_z(i:kz,i);
end

parm_z=parm_z(:); % parm_z=vec(parm_z);
parm(1:kz*(kz*tlz+tly+kzx))=parm_z;
parm(kz*(kz*tlz+tly+kzx)+1:kparm)=a_z;
[pout,like,exitflag,output]=fminsearch('f_sdpd_endo_Wt_persistent',parm,options,yt,ysl,R1t,k1,zt,R2t,W);
results.iter = output.iterations;
results.lik = -t*like; % see f_sar_sdpd_Wt

lambda=0; % restricted under the null

phi2=pout(1:kz*kR2);
Phi2=reshape(phi2,kR2,kz);

vSigma=pout(1+kz*kR2:kparm);
Sigma=zeros(kz,kz);
for i=1:kz
    Sigma(i:kz,i)=vSigma((i-1)*kz+1-(i-1)*(i-2)/2:i*kz-i*(i-1)/2);
end
Sigma=Sigma+Sigma';
for i=1:kz
    Sigma(i,i)=0.5*Sigma(i,i);
    Sigma(i,i)=exp(Sigma(i,i));
end

vSigma=zeros(J,1); % this is for the output of vectorized Sigma (lower half block)
for i=1:kz
    vSigma((i-1)*kz+1-(i-1)*(i-2)/2:i*kz-i*(i-1)/2)=Sigma(i:kz,i);
end

Syt=yt-lambda*ysl;
epsilon=zt-R2t*Phi2;
R1e=R1t(:,[1,3:2+k1]); % Under the null: rho0=0, delta0=0
phi1_ini = R1e\Syt;
phi1 = [phi1_ini(1) 0 phi1_ini(2:length(phi1_ini))']; % rho0=0, restricted under the null
xi=Syt-R1t*phi1';
xipxi = xi'*xi;
sigma=xipxi/L;
delta=zeros(p,1);  % restricted under the null

% Restricted ML estimates (biased by fixed effects)
results.lambda = lambda;
results.phi1 = phi1; 
results.delta = delta;
results.Phi2 = Phi2; results.phi2 = phi2;
results.Sigma = Sigma; results.vSigma = vSigma;
results.sigma = sigma;
results.theta=[results.lambda;results.phi1';results.delta;results.phi2;results.sigma;vSigma];
theta=results.theta; [nvar,junk]=size(theta);


% Information matrix estimator
trG=0;
trG2=0;
for i=1:t
    Wt=W(:,1+i*n:(i+1)*n);
    St=speye(n)-lambda*Wt;
    Gt=Wt*inv(St);
    trG=trG+trace(Gt);
    trG2=trG2+trace(Gt^2);
end

J=kz*(kz+1)/2;
Sigmai=inv(Sigma);
ISigma=zeros(J,J); % this is to compute the information matrix of the Sigma component

for k=1:J
    Sigmak=zeros(kz,kz);
    %ii=floor(solve('ii*kz-(ii-1)*ii/2=k','ii<kz'));
    ii=1.5+kz-sqrt((1.5+kz)^2-2*(k+1));
    ii1=floor(ii);ii2=ceil(ii);
    Sigmak(k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1,ii2)=1;
    if k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1~=ii2
        Sigmak=Sigmak+Sigmak';
    end

    for j=1:J
        Sigmaj=zeros(kz,kz);
        %ii=floor(solve('ii*kz-(ii-1)*ii/2=j','ii<kz'));
        ii=1.5+kz-sqrt((1.5+kz)^2-2*(j+1));
        ii1=floor(ii);ii2=ceil(ii);
        Sigmaj(j-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1,ii2)=1;
        if j-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1~=ii2
        Sigmaj=Sigmaj+Sigmaj';
        end

        ISigma(k,j)=0.5*L*trace(Sigmai*Sigmak*Sigmai*Sigmaj);
    end
end

xpx = zeros(nvar,nvar);
xpx(1,1)=(1/sigma)*ysl'*ysl+trG2; % lambda
xpx(2:kR1+1,1)=(1/sigma)*R1t'*ysl; % phi1
xpx(kR1+2:1+kR1+kz,1)=(1/sigma)*epsilon'*ysl; % delta
xpx(2+kR1+kz:1+kR1+kz+kz*kR2,1)=-(1/sigma)*kron(delta,R2t'*ysl); % phi2
xpx(2+kR1+kz+kz*kR2,1)=(1/sigma)*trG; % sigma
xpx(3+kR1+kz+kz*kR2:2+kR1+kz+kz*kR2+J,1)=zeros(J,1); % Sigma
xpx(1,2:kR1+1)=xpx(2:kR1+1,1)';
xpx(1,kR1+2:1+kR1+kz)=xpx(kR1+2:1+kR1+kz,1)';
xpx(1,2+kR1+kz:1+kR1+kz+kz*kR2)=xpx(2+kR1+kz:1+kR1+kz+kz*kR2,1)';
xpx(1,2+kR1+kz+kz*kR2)=xpx(2+kR1+kz+kz*kR2,1)';
xpx(1,3+kR1+kz+kz*kR2:2+kR1+kz+kz*kR2+J)=xpx(3+kR1+kz+kz*kR2:2+kR1+kz+kz*kR2+J,1)';

xpx(2:kR1+1,2:kR1+1)=(1/sigma)*R1t'*R1t; % phi1
xpx(2+kR1+kz:1+kR1+kz+kz*kR2,2:kR1+1)=-(1/sigma)*kron(delta,R2t'*R1t); % phi2
xpx(2:kR1+1,2+kR1+kz:1+kR1+kz+kz*kR2)=xpx(2+kR1+kz:1+kR1+kz+kz*kR2,2:kR1+1)';

xpx(kR1+2:1+kR1+kz,kR1+2:1+kR1+kz)=(1/sigma)*epsilon'*epsilon; % delta

xpx(2+kR1+kz:1+kR1+kz+kz*kR2,2+kR1+kz:1+kR1+kz+kz*kR2)=(1/sigma)*kron((sigma*inv(Sigma)+delta*delta'),R2t'*R2t); % phi2

xpx(2+kR1+kz+kz*kR2,2+kR1+kz+kz*kR2)=(1/sigma^2)*(0.5*L-t-n+1); % sigma

xpx(3+kR1+kz+kz*kR2:2+kR1+kz+kz*kR2+J,3+kR1+kz+kz*kR2:2+kR1+kz+kz*kR2+J)=ISigma; % Sigma

xpx=xpx/L;
xpxi = invpd(xpx);


% Bias corrected ML estimates
In=eye(n);

Bias1=zeros(nvar,1); % from individual effects
Bias2=zeros(nvar,1); % from time effects

S=kron(ones(1,t+1),In)-lambda*W;%first S is time period 0
G=zeros(n,n*t);
for s=1:t
    G(:,1+(s-1)*n:s*n)=W(:,1+s*n:(s+1)*n)*inv(S(:,1+s*n:(s+1)*n));
end
A=zeros(n,n*t); % first A is time period 1

sumG=zeros(n,n);
for s=1:t
    sumG=sumG+ G(:,1+(s-1)*n:s*n);
end

for s=1:t
    if stl + tl == 2
        gamma_coff=phi1(1);rho_coff=phi1(2);
        A(:,1+(s-1)*n:s*n)=inv(S(:,1+s*n:(s+1)*n))*(gamma_coff*In+rho_coff*W(:,1+(s-1)*n:s*n));
    elseif stl + tl == 1
            if stl == 1
            rho_coff=phi1(1);
            A(:,1+(s-1)*n:s*n)=inv(S(:,1+s*n:(s+1)*n))*(rho_coff*W(:,1+(s-1)*n:s*n));
            else
            gamma_coff=phi1(1);
            A(:,1+(s-1)*n:s*n)=inv(S(:,1+s*n:(s+1)*n))*(gamma_coff*In);
            end
    elseif stl + tl == 0
            error('Wrong Info input,Our model has dynamic term anyway');
    else
            error('Double-Check stl & tl # in Info structure ');
    end
end

bias=zeros(1+tl+stl,1);
bias1=0;bias2=0;bias3=0;
if stl + tl == 2
    for s=1:t-1
        temp1a=eye(n); % this is actually h=1
        temp2a=W(:,1+s*n:(s+1)*n);
        temp3a=G(:,1+s*n:(s+1)*n)*(gamma_coff*temp1a+rho_coff*temp2a);
        for h=2:t-s
            temp1b=eye(n);
            for j=1:h-1
                temp1b=temp1b*A(:,1+(s+j-1)*n:(s+j)*n);
            end
            temp1a=temp1a+temp1b;
            temp2a=temp2a+W(:,1+(s+h-1)*n:(s+h)*n)*temp1b;
            temp3a=temp3a+gamma_coff*G(:,1+(s+h-1)*n:(s+h)*n)*temp1b+rho_coff*G(:,1+(s+h-1)*n:(s+h)*n)*W(:,1+(s+h-1)*n:(s+h)*n)*temp1b;
        end
        bias1=bias1+trace(inv(S(:,1+s*n:(s+1)*n))*temp1a);
        bias2=bias2+trace(inv(S(:,1+s*n:(s+1)*n))*temp2a);
        bias3=bias3+trace(inv(S(:,1+s*n:(s+1)*n))*temp3a);
    end

    bias(1,1)=(1/L)*bias3+(1/L)*trG;
    bias(2,1)=(1/L)*bias1;
    bias(3,1)=(1/L)*bias2;

elseif stl + tl == 1  
        if stl == 1
            for s=1:t-1
                temp1a=eye(n);%this is actually h=1
                temp2a=W(:,1+s*n:(s+1)*n);
                temp3a=G(:,1+s*n:(s+1)*n)*rho_coff*temp2a;
            for h=2:t-s
                temp1b=eye(n);
                for j=1:h-1
                    temp1b=temp1b*A(:,1+(s+j-1)*n:(s+j)*n);
                end
                temp2a=temp2a+W(:,1+(s+h-1)*n:(s+h)*n)*temp1b;
                temp3a=temp3a+rho_coff*G(:,1+(s+h-1)*n:(s+h)*n)*W(:,1+(s+h-1)*n:(s+h)*n)*temp1b;
            end
            bias2=bias2+trace(inv(S(:,1+s*n:(s+1)*n))*temp2a);
            bias3=bias3+trace(inv(S(:,1+s*n:(s+1)*n))*temp3a);
            end

            bias(1,1)=(1/L)*bias3+(1/L)*trG;
            bias(2,1)=(1/L)*bias2;

        else
            for s=1:t-1
                temp1a=eye(n); % this is actually h=1
                temp2a=W(:,1+s*n:(s+1)*n);
                temp3a=G(:,1+s*n:(s+1)*n)*gamma_coff*temp1a;
                for h=2:t-s
                    temp1b=eye(n);
                    for j=1:h-1
                        temp1b=temp1b*A(:,1+(s+j-1)*n:(s+j)*n);
                    end
                    temp1a=temp1a+temp1b;
                    temp3a=temp3a+gamma_coff*G(:,1+(s+h-1)*n:(s+h)*n)*temp1b;
                end
                bias1=bias1+trace(inv(S(:,1+s*n:(s+1)*n))*temp1a);
                bias3=bias3+trace(inv(S(:,1+s*n:(s+1)*n))*temp3a);
            end

            bias(1,1)=(1/L)*bias3+(1/L)*trG;
            bias(2,1)=(1/L)*bias1;


        end
    elseif stl + tl == 0
        error('Wrong Info input,Our model has dynamic term anyway');
else
        error('Double-Check stl & tl # in Info structure ');
end

Bias1(1:1+tl+stl,1)=bias;

bias=zeros(kz*tlz+tly+kzx,kz); % the phi1 componeL bias is gone and we now compute phi2 componeL bias, we will vector it
bias1=zeros(kz*tlz,kz);bias2=zeros(tly,kz);

if tlz + tly == 2
    Phi2z=Phi2(1:kz,:); Phi2y=Phi2(kz+1,:);
    for s=1:t-1
        temp1a=eye(kz);
        temp2a=eye(n);
        for h=2:t-s
            temp1b=Phi2z'.^h;
            temp2b=eye(n);
            for j=1:h-1
                temp2b=temp2b*A(:,1+(s+j-1)*n:(s+j)*n);
            end
            temp1a=temp1a+temp1b;
            temp2a=temp2a+temp2b;
        end
        bias1=bias1+temp1a;
        bias2=bias2+trace(inv(S(:,1+s*n:(s+1)*n))*temp2a)*delta';

    end
    bias(1:kz*tlz+tly,:)=[(1/t)*bias1;-(1/L)*bias2];
    %bias=vec(bias);   
    bias=reshape(bias,[],1);

elseif tlz + tly == 1
    if tlz == 1
        Phi2z=Phi2(1:kz,:);
        for s=1:t-1
            temp1a=eye(kz);
            for h=2:t-s
                temp1b=Phi2z'.^h;
                temp1a=temp1a+temp1b;
            end
            bias1=bias1+temp1a;
        end
        bias(1:kz,:)=(1/t)*bias1;
        %bias=vec(bias);
        bias=reshape(bias,[],1);

    else
        Phi2y=Phi2(1,:);
        for s=1:t-1
            temp2a=eye(n);
            for h=2:t-s
                temp2b=eye(n);
            for j=1:h-1
                temp2b=temp2b*A(:,1+(s+j-1)*n:(s+j)*n);
            end
            temp2a=temp2a+temp2b;
            end

        bias2=bias2+trace(inv(S(:,1+s*n:(s+1)*n))*temp2a)*delta';

        end
        bias(1,:)=-(1/L)*bias2;
        %bias=vec(bias);
        bias=reshape(bias,[],1);
    end
end

Bias1(2+kR1+kz:1+kR1+kz+kR2*kz,1)=bias; % this is bias from Phi2
Bias1(2+kR1+kz+kR2*kz,1)=0.5*inv(sigma);

bias=zeros(J,1); % this is the bias for vSigma
for k=1:J
    Sigmak=zeros(kz,kz);
    %ii=floor(solve('ii*kz-(ii-1)*ii/2=k','ii<kz'));
    ii=1.5+kz-sqrt((1.5+kz)^2-2*(k+1));
    ii1=floor(ii);ii2=ceil(ii);
    Sigmak(k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1,ii2)=1;
    if k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1~=ii2
        Sigmak=Sigmak+Sigmak';
    end
    bias(k,1)=0.5*trace(Sigmai*Sigmak);
end

Bias1(3+kR1+kz+kR2*kz:2+kR1+kz+kR2*kz+J,1)=bias;

Bias2(1,1)=(1/L)*ones(1,n)*sumG*ones(n,1);
Bias2(2+kR1+kz+kR2*kz,1)=0.5*inv(sigma);
Bias2(3+kR1+kz+kR2*kz:2+kR1+kz+kR2*kz+J,1)=bias;              

theta1=theta+xpxi*Bias1/t+xpxi*Bias2/n; % a_{1,\theta0}=Bias1; a_{2,\theta0}=Bias2;

lambda1=0; % restricted under the null
phi11=theta1(2:kR1+1,1); phi11(2,1)=0; % restricted under the null: rho0=0
delta1=zeros(p,1); % restricted under the null
phi21=theta1(kR1+2+kz:kR1+1+kz+kR2*kz,1);
sigma1=theta1(kR1+2+kz+kR2*kz,1);
vSigma1=theta1(kR1+3+kz+kR2*kz:kR1+2+kz+kR2*kz+J,1);

Sigma1=zeros(kz,kz);
for i=1:kz
    Sigma1(i:kz,i)=vSigma1((i-1)*kz+1-(i-1)*(i-2)/2:i*kz-i*(i-1)/2);
end
Sigma1=Sigma1+Sigma1';
for i=1:kz
    Sigma1(i,i)=0.5*Sigma1(i,i);
end

Phi21=reshape(phi21,kR2,kz);
lambda=lambda1;
phi1=phi11;
delta=delta1;
phi2=phi21;
sigma=sigma1;
Sigma=Sigma1;vSigma=vSigma1;
Phi2=Phi21;

results.lambda1 = lambda;
results.phi11 = phi1; gamma=phi1(1); rho=0; % Under the null: rho=0
results.delta1 = delta;
results.Phi21 = Phi2;
results.Sigma1 = Sigma;
results.sigma1 = sigma;
results.theta=[results.lambda1;results.phi11;results.delta1; ...
                reshape(results.Phi21,[],1);results.sigma1;reshape(results.Sigma1,[],1)];
theta=results.theta;

% Unbiased residuals
epsilon=zt-R2t*Phi2;
xi=Syt-R1t*phi1;

% Unbiased quantities
S_L=eye(L)-lambda*W1-gamma*W2-rho*W3;
G1=W1/S_L;
G2=W2/S_L;
G3=W3/S_L;

% Unbiased estimators for the information matrix (at restricted ML)
I_lambdalambda=ysl'*ysl+sigma*trG2; % lambda

I_phi1lambda=R1t'*ysl; % phi1
I_gammalambda=I_phi1lambda(1);
I_rholambda=I_phi1lambda(2);
I_betalambda=I_phi1lambda(3:3+k1-1);

I_deltalambda=epsilon'*ysl; % delta
tmp=epsilon'*R1t;
I_deltagamma=tmp(:,1);
I_deltarho=tmp(:,2);
I_sigmalambda=trG; % sigma

I_phi1phi1=R1t'*R1t; % phi1
I_gammagamma=I_phi1phi1(1,1);
I_gammarho=I_phi1phi1(1,2);
I_rhorho=I_phi1phi1(2,2);
I_gammabeta=I_phi1phi1(1,3:3+k1-1);
I_rhobeta=I_phi1phi1(2,3:3+k1-1);
I_betabeta=I_phi1phi1(3:3+k1-1,3:3+k1-1);

I_phi2phi2=kron(sigma*inv(Sigma),R2t'*R2t); % phi2
I_sigmasigma=(1/sigma)*(0.5*L-t-n+1); % sigma

I_deltadelta=epsilon'*epsilon/(L*sigma); % delta
I_deltaomega=zeros(p,k1+2); %
I_deltaeta=[I_deltalambda I_deltarho]/(L*sigma); %
I_omegaomega=[I_gammagamma I_gammabeta zeros(1,1); ...
              I_gammabeta' I_betabeta zeros(k1,1); ...
              zeros(1,1) zeros(k1,1)' I_sigmasigma]/(L*sigma); %
I_etaomega=[I_gammalambda' I_betalambda' I_sigmalambda'; ...
            I_gammarho' I_rhobeta 0]/(L*sigma); %                       
I_etaeta=[I_lambdalambda I_rholambda'; ...
          I_rholambda I_rhorho]/(L*sigma); %

I_jj=[I_deltadelta I_deltaeta;
        I_deltaeta' I_etaeta]; % 'j' stands for 'joint'
I_jomega=[I_deltaomega; I_etaomega];

% Bias terms
D_1delta=zeros(p,1); %
D_2delta=zeros(p,1); %

D_1gamma=trace(G2*kron(ones(T,1)*ones(T,1)'/T,Jn));
%D_1gamma=sqrt(n/T)*(1/(n-1))*trace(G2*kron(ones(T,1)*ones(T,1)'/T,Jn));
D_1beta=zeros(k1,1);
D_1sigma=(n-1)/sqrt(L)*(-1/(2*sigma));
%D_1sigma=sqrt(n/T)*(-1/(2*sigma));
D_1omega=[D_1gamma; D_1beta; D_1sigma]; %

D_2gamma=0;
D_2beta=zeros(k1,1);
D_2sigma=sqrt(T/n)*(-1/(2*sigma));
D_2omega=[D_2gamma; D_2beta; D_2sigma]; %

D_1lambda=-trace(G1*kron(ones(T,1)*ones(T,1)'/T,Jn));
D_1rho=trace(G3*kron(ones(T,1)*ones(T,1)'/T,Jn));
%D_1lambda=sqrt(n/T)*(-1/(n-1))*trace(W1*kron(ones(T,1)*ones(T,1)'/T,Jn));
%D_1rho=sqrt(n/T)*(1/(n-1))*trace(G3*kron(ones(T,1)*ones(T,1)'/T,Jn));
D_1eta=[D_1lambda; D_1rho]/sqrt(L); %

D_2lambda=-trace(G1*kron(speye(T),ones(n,1)*ones(n,1)'/n));
D_2rho=0;
D_2eta=[D_2lambda; D_2rho]/sqrt(L); %


% Score functions (bias-corrected)
L_delta=epsilon'*xi/(L*sigma); %

RJXi=R1t'*xi;
L_lambda=xi'*ysl-sigma*trG;
L_rho=RJXi(2,1);
L_eta=[L_lambda; L_rho]/(L*sigma); %

% Adjusting factors
I_deo=I_deltaeta; % I_delta eta.omega
I_eo=I_etaeta-I_etaomega*(I_omegaomega\I_etaomega'); % I_eta.omega
I_do=I_deltadelta; % I_delta.omega

I_jo=I_jj-I_jomega*(I_omegaomega\I_jomega'); % I_joint.omega

AF=I_deo*inv(I_eo); % Adjusting Factor (AF)

% Test statistics
L_delta_adjusted=sqrt(L)*L_delta-AF*sqrt(L)*L_eta;

C_delta=sqrt(L)*(L_delta-(D_1delta+D_2delta)/sqrt(L)+I_deltaomega*(I_omegaomega\(D_1omega+D_2omega))/sqrt(L));
C_eta=sqrt(L)*(L_eta-(D_1eta+D_2eta)/sqrt(L)+I_etaomega*(I_omegaomega\(D_1omega+D_2omega))/sqrt(L));
C_delta_adjusted=C_delta-AF*C_eta; % 

Var_adjusted=I_do-AF*I_deo'; %

% Report
%RS_biased_paramissp_delta=(sqrt(L)*L_delta)'*inv(I_do)*(sqrt(L)*L_delta);
%RS_biased_delta=L_delta_adjusted'*inv(Var_adjusted)*L_delta_adjusted;
%RS_paramissp_delta=C_delta'*inv(I_do)*C_delta;
RS_robust_delta=C_delta_adjusted'*inv(Var_adjusted)*C_delta_adjusted


