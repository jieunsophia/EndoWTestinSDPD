clear all; clc; warning off;
rng(2024); eps=10^(-4);

% = Experimental inputs ===================================================

% Testing
type1error=0.05; 

% Error distribution
rng_type=1; t_df=0; % rng_type: 1 (normal)
%rng_type=2; t_df=20; % rng_type: 2 (t-dist)

% Asymptotic
nT_set=[100 50]; [length_nT,~]=size(nT_set);

% Simulation setup
sim=1000; 

% Parameters
deltaset=0;
[nrow_deltaset,p]=size(deltaset); critic1=chi2inv((1-type1error),p);

kx1=1; beta0=1;

corr_ife=0.2; corr_tfe=0.1; % Correlation for individual/time fixed effects
corr_z=eye(p); corr_z_tmp=0.3*(ones(p)-eye(p)); corr_z=corr_z+corr_z_tmp;

sig2_v=1;

%tmp=length(0.05:0.05:0.3);
%etaset=zeros(tmp*3,3); % eta = [lambda, gamma, rho]
%for i=1:3
%    etaset(1+(i-1)*tmp:i*tmp,i)=[0.05:0.05:0.3]';
%end
%etaset=zeros(1+tmp*3,1)+[zeros(1,3); etaset];
etaset=[0.3 0 0]; [num_hypothesis_eta , ~]=size(etaset);

kappa0=0.2*ones(p);
kx2=1; Gamma0=0.3*ones(kx2,p);

if p==1
    alpha0=1;
else 
    alpha0=0.25;
end

% =========================================================================

% Step 1. Data generating process for Y0, Z0, and fixed effects
for nT_set_i=1:length_nT
    n=nT_set(nT_set_i,1); T=nT_set(nT_set_i,2);
    L=n*T; nt=L; 

    % Deterministic random variables: X, Y0, Z0, fixed effects
    rng(2022)
    Y0=normrnd(0,1,[n 1]); % Initial values observed
    Z0=mvnrnd(0*ones(p,1),corr_z,n); % Initial values observed

    % X1,X2
    x1=mvnrnd(zeros(kx1+2,1), [eye(kx1) corr_ife*ones(kx1,1) corr_tfe*ones(kx1,1); ...
                              corr_ife*ones(1,kx1) 1 0; ...
                              corr_tfe*ones(1,kx1) 0 1], L);
    X1=x1(:,(1:kx1)); c10=x1(:,(kx1+1)); a10=x1(:,(kx1+2));

    x2=mvnrnd(zeros(kx2+2*p,1), [eye(kx2) corr_ife*ones(p,kx2)' corr_tfe*ones(p,kx2)'; ...
                                corr_ife*ones(p,kx2) eye(p) zeros(p,p); ...
                                corr_tfe*ones(p,kx2) zeros(p,p) eye(p)], L);
    X2=x2(:,(1:kx2)); c20=x2(:,(kx2+1):(kx2+p)); a20=x2(:,(kx2+p+1):(kx2+p+p));

    
    % Individual fixed effects (c1,c2)
    mean_c1=[]; 
    for i=1:n
        c12=[]; c22=[];
        for j=1:T
            c11=c10((j-1)*n+i); c12=[c12 c11]; 
        end
        mean_c1=[mean_c1 mean(c12)];
    end
    c1=kron(ones(T,1),mean_c1');
    
    mean_c2=[];
    for i=1:n
        c22=[];
        mean_c2p=[];
        for jj=1:p
            c22=[];
            for j=1:T
                c21=c20((j-1)*n+i,jj);
                c22=[c22 c21];
            end
            mean_c2p=[mean_c2p mean(c22)];
        end
        mean_c2=[mean_c2; mean_c2p];
    end
    c2=kron(ones(T,1),mean_c2); 

    
    % Time fixed effects (a1,a2)
    mean_a1=[]; 
    for t=1:T
        a1_vec=a10((t-1)*n+1:n*t);
        mean_a1=[mean_a1 mean(a1_vec)];
    end
    a1=kron(mean_a1',ones(n,1));
    
    mean_a2=[];
    for t=1:T
        mean_a2p=[];
        for jj=1:p
            a2_vec=a20((t-1)*n+1:n*t,jj);
            mean_a2p=[mean_a2p mean(a2_vec)];
        end
        mean_a2=[mean_a2; mean_a2p];
    end
    a2=kron(mean_a2,ones(n,1)); 

    
    twoway_fe_corr=[corr(X1,c1)' corr(X1,a1)' ...
                    reshape(corr(X2,c2),[],1)' reshape(corr(X2,a2),[],1)']; % nonzero 
    disp('Norm of twoway_fe_corr:'); norm(twoway_fe_corr)

    % Spatial weight matrices: exogenous part
    for w_type=1:2
        if w_type==1
            W_d=W_Queen(n); W_type='Queen'; % Time-invariant physical distance (W by Chess Queen rule)
        else
            W_d=W_Rook(n); W_type='Rook'; % W by Chess Rook rule
        end
        W_dd=kron(eye(T),W_d);

        % W0
        W0_e=zeros(n,n);
        for i=1:n
            for j=i+1:n
                W0_e(i,j)=1/norm(Z0(i,:)-Z0(j,:));
            end
        end
        W0_e=W0_e+W0_e';    
        W0=normw(W_d.*W0_e);            

        identity=[];
        % = Step2. Data generating process for Y, Z, X1, X2 ===============
        for delta_i=1:nrow_deltaset % Loop for delta
            delta0=deltaset(delta_i,:)';
            
            % Declare \sigma_{\xi 0}^{2} (Codes notation: xi0)
            if p==1
                Sig_e=alpha0;
                sig_ve=delta0*Sig_e;
                xi0=sig2_v-sig_ve^2/alpha0;
                disp('Corr between v and e:'); sig_ve/sqrt(sig2_v*alpha0)
            else
                Sig_e=eye(p)+alpha0*(ones(p)-eye(p));
                sig_ve=Sig_e*delta0; % p by 1
                xi0=sig2_v-sig_ve'/Sig_e*sig_ve;
                disp('Corr between v and e:'); (sqrt(sig2_v)*sqrt(Sig_e))\sig_ve
            end
            
            for eta_i=1:num_hypothesis_eta % Loop for eta
                eta0=etaset(eta_i,:); lambda0=eta0(1); gamma0=eta0(2); rho0=eta0(3);

                RejectionRate_standardRS=[]; 
                RejectionRate_biased_paramissp_RS=[]; RejectionRate_biasedRS=[]; RejectionRate_paramisspRS=[]; RejectionRate_robustRS=[];
                tElapsed_RobustRS=[];
                for sim_i=1:sim % Loop for simulation
                    tStart=tic;
                    % Seed number (variant)
                    rng(sim_i*7+12);

                    % Error terms
                    if rng_type==1
                        tmp=mvnrnd(zeros(1+p,1),[sig2_v sig_ve';sig_ve Sig_e],L); % Normal
                    else
                        tmp=mvtrnd([sig2_v sig_ve';sig_ve Sig_e],t_df,L); % t-dist (Fat tails)
                    end
                    v=tmp(:,1); e=tmp(:,2:(1+p));

                    % Generate Z
                    Z=[];
                    for t=1:T
                        if t==1
                            Z(1:n,:)=Z0*kappa0+X2(1:n,:)*Gamma0+c2(1:n,:)+a2(1:n,:)+e(1:n,:);
                        else
                            Z((t-1)*n+1:t*n,:)=Z((t-2)*n+1:(t-1)*n,:)*kappa0...
                                                +X2((t-1)*n+1:t*n,:)*Gamma0...
                                                +c2((t-1)*n+1:t*n,:)+a2((t-1)*n+1:t*n,:)...
                                                +e((t-1)*n+1:t*n,:);
                        end
                    end
                    Z_stack=[Z0; zeros(n*T,p)]+[zeros(n,p); Z];


                    % - Generate W_e & W1.2.3 -----------------------------
                    W_e=zeros(L,L); % W at t>=1, W symmetric
                    for t=1:T % (t,t)
                        for i=1:n
                            for j=(t-1)*n+(i+1):(t-1)*n+n
                            W_e((t-1)*n+i,j)=1/norm(Z((t-1)*n+i,:)-Z(j,:));
                            end
                        end
                            W_e((t-1)*n+1:t*n,(t-1)*n+1:t*n)=W_e((t-1)*n+1:t*n,(t-1)*n+1:t*n)+W_e((t-1)*n+1:t*n,(t-1)*n+1:t*n)';
                    end


                    % W = lambda0*W1 + gamma0*W2 + rho0*W3
                    W1=normw(W_dd.*W_e);
                    W2=zeros(L,L); W3=zeros(L,L);
                    for t=1:T-1
                        W2(n*t+1:n*(t+1),(t-1)*n+1:t*n)=eye(n);
                        W3(n*t+1:n*(t+1),(t-1)*n+1:t*n)=W1((t-1)*n+1:t*n,(t-1)*n+1:t*n);
                    end

                    W=lambda0*W1+gamma0*W2+rho0*W3;


                    % Generate Y
                    ell=zeros(L,1); ell(1:n)=gamma0*Y0+rho0*W0*Y0;

                    JT=speye(T)-ones(T,1)*ones(T,1)'/T; Jn=speye(n)-ones(n,1)*ones(n,1)'/n;
                    JL=kron(JT,Jn);
                    SL=speye(L)-W;

                    Y=SL\(X1*beta0+ell+c1+a1+v);
                    Y_stack=[Y0' zeros(n*T,1)']'+[zeros(n,1)' Y']';

                    % Rename variables
                    w=zeros(n,n*(T+1));
                    for t=1:T+1
                        if t==1
                            w(:,1:n)=W0;
                        else  w(:,(t-1)*n+1:t*n)=W1((t-2)*n+1:(t-1)*n,(t-2)*n+1:(t-1)*n);
                        end
                    end

                    y=Y_stack; x=X1; z=Z_stack; zx=X2; W=w;           

                    info = struct('n',n,'t',t,'rmin',0,'rmax',1,'lflag',0,'tl',1,'stl',1,'tlz',1,'tly',0);            
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
                    options=optimset('fminsearch');
                    options=optimset(options,'MaxFunEvals',100000);
                    options=optimset(options,'MaxIter',100000);
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
                    [pout,like,exitflag,output]=fminsearch('f_sdpd_endo_Wt_restricted',parm,options,yt,ysl,R1t,kx1,zt,R2t,W); % restricted likelihood function under the null: lambda0=gamma0=rho0=0
                    results.iter = output.iterations;
                    results.lik = -t*like; % see f_sar_sdpd_Wt

                    lambda=0; % Under the null: lambda=0

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
                    R1e=R1t(:,(3:(2+kx1))); % Under the null: gamma=rho=0, delta=0
                    phi1_beta = R1e\Syt;
                    phi1 = [0 0 phi1_beta']'; % Under the null: gamma=rho=0
                    xi=Syt-R1t*phi1;
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
                    results.theta=[results.lambda results.phi1' results.delta' ...
                                   results.phi2' results.sigma vSigma']';
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
                        bias=reshape(bias,[],1); % Manually vectorize (For lower version Matlab)

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
                            bias=reshape(bias,[],1); % Manually vectorize (For lower version Matlab)

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
                            bias=reshape(bias,[],1); % Manually vectorize (For lower version Matlab)
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
                    phi11=[0 0 theta1(4:(3+kx1),1)']'; % restricted under the null: gamma0=rho0=0
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
                    results.phi11 = phi1;
                    results.delta1 = delta;
                    results.Phi21 = Phi2; results.phi21 = phi21;
                    results.Sigma1 = Sigma;
                    results.sigma1 = sigma;
                    results.vSigma1=vSigma1;
                    results.theta=[results.lambda1;results.phi11;results.delta1;results.phi21;results.sigma1;results.vSigma1];
                    theta=results.theta;

                    % Unbiased residuals
                    %Syt=yt-lambda*ysl; % Same as before because lambda=0 (restricted)
                    epsilon=zt-R2t*Phi2;
                    xi=Syt-R1t*phi1;

                    % Unnecessary process (because they are same as before with restricted lambda=0)
                    %[nvar,junk]=size(theta);
                    %trG=0;
                    %trG2=0;
                    %for i=1:t
                    %    St=speye(n)-lambda*W(:,1+i*n:(i+1)*n);
                    %    Wt=W(:,1+i*n:(i+1)*n);
                    %    Gt=Wt*inv(St);
                    %    trG=trG+trace(Gt);
                    %    trG2=trG2+trace(Gt^2);
                    %end

                    % Unnecessary process (because alpha or Sigma is not really considered as info matrix forms block-diagonal w.r.t. alpha)
                    %J=kz*(kz+1)/2;
                    %Sigmai=inv(Sigma);
                    %ISigma=zeros(J,J);

                    %for k=1:J
                    %    Sigmak=zeros(kz,kz);
                        %ii=floor(solve('ii*kz-(ii-1)*ii/2=k','ii<kz'));
                    %    ii=1.5+kz-sqrt((1.5+kz)^2-2*(k+1));
                    %    ii1=floor(ii);ii2=ceil(ii);
                    %    Sigmak(k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1,ii2)=1;
                    %    if k-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1~=ii2
                    %        Sigmak=Sigmak+Sigmak';
                    %    end

                    %    for j=1:J
                    %        Sigmaj=zeros(kz,kz);
                            %ii=floor(solve('ii*kz-(ii-1)*ii/2=j','ii<kz'));
                    %        ii=1.5+kz-sqrt((1.5+kz)^2-2*(j+1));
                    %        ii1=floor(ii);ii2=ceil(ii);
                    %        Sigmaj(j-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1,ii2)=1;
                    %        if j-(ii1*kz-(ii1-2)*(ii1-1)/2)+ii1~=ii2
                    %            Sigmaj=Sigmaj+Sigmaj';
                    %        end
                    %        ISigma(k,j)=0.5*nt*trace(Sigmai*Sigmak*Sigmai*Sigmaj);
                    %    end
                    %end
                    
                    
                    % Unbiased estimators for the information matrix (at restricted ML)
                    I_lambdalambda=ysl'*ysl+sigma*trG2; % lambda

                    I_phi1lambda=R1t'*ysl; % phi1
                    I_gammalambda=I_phi1lambda(1);
                    I_rholambda=I_phi1lambda(2);
                    I_betalambda=I_phi1lambda(3:(3+kx1-1));

                    I_deltalambda=epsilon'*ysl; % delta
                    tmp=epsilon'*R1t;
                    I_deltagamma=tmp(:,1);
                    I_deltarho=tmp(:,2);
                    I_sigmalambda=trG; % sigma

                    I_phi1phi1=R1t'*R1t; % phi1
                    I_gammagamma=I_phi1phi1(1,1);
                    I_gammarho=I_phi1phi1(1,2);
                    I_rhorho=I_phi1phi1(2,2);
                    I_gammabeta=I_phi1phi1(1,3:(3+kx1-1));
                    I_rhobeta=I_phi1phi1(2,3:(3+kx1-1));
                    I_betabeta=I_phi1phi1(3:(3+kx1-1),3:(3+kx1-1));

                    I_phi2phi2=kron(sigma*inv(Sigma),R2t'*R2t); % phi2
                    I_sigmasigma=(1/sigma)*(0.5*L-t-n+1); % sigma

                    I_deltadelta=epsilon'*epsilon/(L*sigma); % delta
                    I_deltaomega=zeros(p,(kx1+1)); %
                    I_deltaeta=[I_deltalambda I_deltagamma I_deltarho]/(L*sigma); %
                    I_omegaomega=[I_betabeta zeros(kx1,1); zeros(1,kx1) I_sigmasigma]/(L*sigma); %
                    I_etaomega=[I_betalambda' I_sigmalambda'; I_gammabeta 0; I_rhobeta 0]/(L*sigma); %                         
                    I_etaeta=[I_lambdalambda I_gammalambda' I_rholambda';...
                                  I_gammalambda I_gammagamma I_gammarho;...
                                  I_rholambda I_gammarho' I_rhorho]/(L*sigma); %

                    I_jj=[I_deltadelta I_deltaeta;
                            I_deltaeta' I_etaeta]; % 'j' stands for 'joint'
                    I_jomega=[I_deltaomega; I_etaomega];

                    % Bias terms
                    D_1delta=zeros(p,1); %
                    D_2delta=zeros(p,1); %

                    D_1beta=zeros(kx1,1);
                    D_1sigma=(n-1)/sqrt(L)*(-1/(2*sigma));
                    %D_1sigma=sqrt(n/T)*(-1/(2*sigma));
                    D_1omega=[D_1beta; D_1sigma]; %

                    D_2beta=zeros(kx1,1);
                    D_2sigma=sqrt(T/n)*(-1/(2*sigma));
                    D_2omega=[D_2beta; D_2sigma]; %

                    D_1lambda=-trace(W1*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    D_1gamma=trace(W2*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    D_1rho=trace(W3*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    %D_1lambda=sqrt(n/T)*(-1/(n-1))*trace(W1*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    %D_1gamma=sqrt(n/T)*(1/(n-1))*trace(W2*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    %D_1rho=sqrt(n/T)*(1/(n-1))*trace(W3*kron(ones(T,1)*ones(T,1)'/T,Jn));
                    D_1eta=[D_1lambda; D_1gamma; D_1rho]/sqrt(L); %

                    D_2lambda=-trace(W1*kron(speye(T),ones(n,1)*ones(n,1)'/n));
                    D_2gamma=0;
                    D_2rho=0;
                    D_2eta=[D_2lambda; D_2gamma; D_2rho]/sqrt(L); %


                    % Score functions (bias-corrected)
                    L_delta=epsilon'*xi/(L*sigma); %

                    RJXi=R1t'*xi;
                    L_lambda=xi'*ysl-sigma*trG;
                    L_gamma=RJXi(1,1);
                    L_rho=RJXi(2,1);
                    L_eta=[L_lambda; L_gamma; L_rho]/(L*sigma); %

                    % Adjusting factors
                    I_deo=I_deltaeta; % I_delta eta.omega
                    I_eo=I_etaeta-I_etaomega*(I_omegaomega\I_etaomega'); % I_eta.omega
                    I_do=I_deltadelta; % I_delta.omega

                    I_jo=I_jj-I_jomega*(I_omegaomega\I_jomega'); % I_joint.omega

                    AF=I_deo*inv(I_eo); % Adjusting Factor (AF)

                    % Test statistics
                    L_delta_adjusted=sqrt(L)*L_delta-AF*sqrt(L)*L_eta;
                    C_delta_adjusted=sqrt(L)*L_delta-AF*sqrt(L)*(L_eta-(D_1eta+D_2eta)/sqrt(L)+I_etaomega*(I_omegaomega\(D_1omega+D_2omega))/sqrt(L)); % 

                    Var_adjusted=I_do-AF*I_deo'; %

                    RS_biased_paramissp_delta=(sqrt(L)*L_delta)'*inv(I_do)*(sqrt(L)*L_delta);
                    RS_biased_delta=L_delta_adjusted'*inv(Var_adjusted)*L_delta_adjusted;
                    RS_paramissp_delta=(sqrt(L)*L_delta)'*inv(I_do)*(sqrt(L)*L_delta);
                    RS_robust_delta=C_delta_adjusted'*inv(Var_adjusted)*C_delta_adjusted;

                    % Equality check
                    %LHS=RS_standard_joint;
                    %RHS=RS_robust_delta + RS_standard_eta;
                    %identity=[identity; [delta0 eta0 LHS RHS (abs(LHS-RHS)<=eps)]];

                    % Record
                    RejectionRate_biased_paramissp_RS=[RejectionRate_biased_paramissp_RS (RS_biased_paramissp_delta>critic1)];
                    RejectionRate_biasedRS=[RejectionRate_biasedRS (RS_biased_delta>critic1)];
                    RejectionRate_paramisspRS=[RejectionRate_paramisspRS (RS_paramissp_delta>critic1)];
                    RejectionRate_robustRS=[RejectionRate_robustRS (RS_robust_delta>critic1)];

                    tElapsed_RobustRS=[tElapsed_RobustRS toc(tStart)];

                    end % Simulation ends

                    % Estimated Rejection Rate (Size or Power) & time elapsed
                    RejectionRateHat_biased_paramissp_RS=sum(RejectionRate_biased_paramissp_RS)/sim;
                    RejectionRateHat_biasedRS=sum(RejectionRate_biasedRS)/sim;
                    RejectionRateHat_paramissp_RS=sum(RejectionRate_paramisspRS)/sim;
                    RejectionRateHat_robustRS=sum(RejectionRate_robustRS)/sim;

                    avg_tElapsed_RobustRS=mean(tElapsed_RobustRS);

                    % Save
                    save(strcat(['BP_rej=',num2str(RejectionRateHat_biased_paramissp_RS),' & B_rej=',num2str(RejectionRateHat_biasedRS),' & P_rej=',num2str(RejectionRateHat_paramissp_RS),' & R_rej=',num2str(RejectionRateHat_robustRS),', avg_elapsed=',num2str(avg_tElapsed_RobustRS),', #sim=',num2str(sim),', n=',num2str(n),', T=',num2str(T),', p=',num2str(p),', delta0=',num2str(delta0'),', lambda0=',num2str(lambda0),', gamma0=',num2str(gamma0),', rho0=',num2str(rho0),', W_type=',W_type,', rng_type=',num2str(rng_type),', t_df=',num2str(t_df),'.mat']));
             end % eta
         end % delta
    end % w_type 
end