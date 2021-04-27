clc
clear
%-----------------Initial image-------------------------%
image1=imread('160_N3F0.bmp');
[row,column]=size(image1);
im1=double(image1);
im1=im1/max(im1(:));
Y=im1(:)';
epsilon=0.001;
%---------------------------------------------------%
%----------------Initialization--------------------------%
h=0.9;
alpha=0.3;
% alpha=0.5;%natual
r=1; %radiu
K=4;%number of class K=4 for brain MR images
    %K=3 for 8143, 134035, 22090, 198023, 22013, 118035, 238011
    %K=6 for 55067
[n,m]=size(Y);%n-Dimension

ROI=ones(row,column);
index = ROI ==1;

yy = im1(index);
yy = sort(yy,'descend');
% K-means for initialization
[IDX,C] = kmeans(yy,K,'Start','cluster', ...
    'Maxiter',100, ...
    'EmptyAction','drop', ...
    'Display','off');
while sum(isnan(C))>0
    [IDX,C] = kmeans(yy,K,'Start','cluster', ...
        'Maxiter',100, ...
        'EmptyAction','drop', ...
        'Display','off');
end
V = sort(C);
Dis_k = zeros(row,column,K);

for k = 1:K
    Dis_k(:,:,k) = (im1 - V(k)).^2;
end
Sigma=zeros(K,1);
Lambda=zeros(n,K)+0.1;%Initialize lambda
%--Initialize Delta,T, delta,Sigma------%
Delta=zeros(n,K);
delta=zeros(n,K);
T=zeros(K,n);
for k = 1:K
    [e_min,IDX] = min(Dis_k,[],3);
    IDX_ROI = IDX.*ROI;
    Sigma(k) = var(im1(IDX_ROI==k));%Initialize Sigma 
    delta(:,k)=Lambda(:,k)/((1+Lambda(:,k)'*Lambda(:,k))^(1/2)+eps);
    Delta(:,k)=sqrt(Sigma(k))*delta(:,k);
    T(k)=Sigma(k)-Delta(:,k)*Delta(:,k)';
end
Mu=V';%Initialize mu
%------------------%
%-------------------%
 S1_ij=zeros(K,m);
 S2_ij=zeros(K,m);
 Pi=ones(m,K)/K;
%% alpha_ij initialization

%-------X=(Xj-Xi)-----------------------------------&
X=zeros(2,9);
X(:,1)=[-1,1]';X(:,2)=[-1,0]';X(:,3)=[-1,-1]';
X(:,4)=[0,1]';X(:,5)=[0,0]';X(:,6)=[0,-1]';
X(:,7)=[1,1]';X(:,8)=[1,0]';X(:,9)=[1,-1]';

Ix=(circshift(im1,[0,-1])-circshift(im1,[0,1]))/2;
Iy=(circshift(im1,1)-circshift(im1,-1))/2;
Ix2=Ix.*Ix;Iy2=Iy.*Iy;Ixy=Ix.*Iy;
L1=0.5*(Ix2+Iy2+sqrt((Ix2-Iy2).^2+4*(Ixy.^2)));
L2=0.5*(Ix2+Iy2-sqrt((Ix2-Iy2).^2+4*(Ixy.^2)));
costhta=2*Ixy;
sinthta=Iy2-Ix2+sqrt((Ix2-Iy2).^2+4*(Ixy.^2));
V_u_1=costhta./(sqrt(costhta.^2+sinthta.^2)+eps);
V_u_2=sinthta./(sqrt(costhta.^2+sinthta.^2)+eps);
Ni=9;
radiu=fix(sqrt(Ni)/2);
index=1;
index_im1=reshape(1:m,[row,column]);
for j=radiu:-1:-radiu
    for i=radiu:-1:-radiu
        temp1=circshift(im1,[i,j]);
        temp2=circshift(V_u_1,[i,j]);
        temp3=circshift(V_u_2,[i,j]);
        temp4=circshift(L1,[i,j]);
        temp5=circshift(L2,[i,j]);
        temp6=circshift(index_im1,[i,j]);
        im1_Ni(:,index)=temp1(:);
        V_u_1_Ni(:,index)=temp2(:);
        V_u_2_Ni(:,index)=temp3(:);
        L1_Ni(:,index)=temp4(:);
        L2_Ni(:,index)=temp5(:);
        index_im1_Ni(:,index)=temp6(:);
        index=index+1;
    end
end
%----------------------w_ij------------------------%
j_label=(Ni+1)-(1:Ni);
center_label=[];
label=index_im1_Ni(index_im1_Ni,j_label);
for j=1:Ni
    center_label=[center_label;label((j-1)*m+1:j*m,j)];
end
wij=abs(im1_Ni(index_im1_Ni,:)-im1_Ni(center_label,:));
wij=sum(wij,2);
wij=reshape(exp(-wij/h),[m Ni]);
temp=wij(:,5);
temp=double(double((wij(:,5)./(sum(wij,2)+eps))>=0.8)==0).*temp;
wij(:,5)=temp;
wij=wij./(repmat(sum(wij,2),1,Ni)+eps);

%----------------------mu_A, mu_Q, lambda1, lambda2, f1, f2,------------------------%
mu_A_1=sum(wij.*V_u_1_Ni,2);
mu_A_2=sum(wij.*V_u_2_Ni,2);
mu_A_1_norm=mu_A_1./(sqrt(mu_A_1.^2+mu_A_2.^2)+eps);
mu_A_2_norm=mu_A_2./(sqrt(mu_A_1.^2+mu_A_2.^2)+eps);
mu_Q_1_norm=-mu_A_2_norm;
mu_Q_2_norm=mu_A_1_norm;
mu_A=[mu_A_1_norm';mu_A_2_norm'];
mu_Q=[mu_Q_1_norm';mu_Q_2_norm'];
lambda1=sum(wij.*L1_Ni,2);
lambda2=sum(wij.*L2_Ni,2);
f_1=1/r^2;f_2=1./(r+lambda1*100+lambda2).^2;

%----------------------D,  alpha_ij------------------------%
D_11=f_1*(mu_A_1_norm.*mu_A_1_norm)+f_2.*(mu_Q_1_norm.*mu_Q_1_norm);
D_12=f_1*(mu_A_1_norm.*mu_A_2_norm)+f_2.*(mu_Q_1_norm.*mu_Q_2_norm);
D_21=f_1*(mu_A_2_norm.*mu_A_1_norm)+f_2.*(mu_Q_2_norm.*mu_Q_1_norm);
D_22=f_1*(mu_A_2_norm.*mu_A_2_norm)+f_2.*(mu_Q_2_norm.*mu_Q_2_norm);
alpha_ij=repmat(X(1,:),m,1).*repmat(D_11,1,Ni).*repmat(X(1,:),m,1)+repmat(X(2,:),m,1).*repmat(D_21,1,Ni).*repmat(X(1,:),m,1)...
    +repmat(X(1,:),m,1).*repmat(D_12,1,Ni).*repmat(X(2,:),m,1)+repmat(X(2,:),m,1).*repmat(D_22,1,Ni).*repmat(X(2,:),m,1);
alpha_ij=exp(-alpha_ij).*wij;
alpha_ij(:,5)=0;
alpha_ij=alpha_ij./(repmat(sum(alpha_ij,2),1,Ni)+eps);



%-----------------------------------------------------------------------------------------------------%
ROI=ROI(:);

%% Parameter update

for t=1:100
    %-------------Update S1_ij,S2_ij-----------------%
    M=1./(sqrt(1+Delta'.*(T.^(-1.0)).*Delta')+eps);
    mu_ij=repmat(Delta'.*(T.^(-1.0))./((1+Delta'.*(T.^(-1)).*Delta')+eps),1,m).*(repmat(Y,K,1)-repmat(Mu',1,m)).*repmat(ROI',K,1);
    S1_ij=mu_ij+(normpdf(mu_ij./(repmat(M,1,m)+eps))./(normcdf(mu_ij./(repmat(M,1,m)+eps))+eps)).*repmat(M,1,m).*repmat(ROI',K,1);
    S2_ij=mu_ij.^2+repmat(M,1,m).^2+(normpdf(mu_ij./(repmat(M,1,m)+eps))./(normcdf(mu_ij./(repmat(M,1,m)+eps))+eps)).*repmat(M,1,m).*mu_ij.*repmat(ROI',K,1);
    %---------------------------------------------------------%
    
    %-------------------Update Z--------------------------%
    
    phi=1./((sqrt(2*pi*repmat(Sigma,1,m)))+eps).*exp(-1/2*((repmat(Y,K,1)-repmat(Mu',1,m)).^2).*(1./(repmat(Sigma,1,m)+eps))).*repmat(ROI',K,1);
    psi=normcdf(repmat(Lambda',1,m).*repmat(Sigma.^(-1.0/2),1,m).*(repmat(Y,K,1)-repmat(Mu',1,m))).*repmat(ROI',K,1);

    Z=2*Pi'.*phi.*psi.*repmat(ROI',K,1);  
    Z=Z./(repmat(sum(Z),K,1)+eps).*repmat(ROI',K,1);
    p1_ik=Z';
    %---------------------------------------------------------%
    
    %--------------------Update p1n_ik,Pin---------------------------%
    
    p1n_ik=zeros(m,K);Pin=zeros(m,K);
    for k=1:K
        p1_ik_reshape=reshape(p1_ik(:,k),[row,column]);  
        Pi_reshape=reshape(Pi(:,k),[row,column]);
        A1=[zeros(row,1),p1_ik_reshape,zeros(row,1)];
        B1=[zeros(1,column+2);A1;zeros(1,column+2)];
        A2=[zeros(row,1),Pi_reshape,zeros(row,1)];
        B2=[zeros(1,column+2);A2;zeros(1,column+2)];
        p1n_ik(:,k)=(sum(alpha_ij'.*im2col(B1, [3,3], 'sliding')))';
        Pin(:,k)=(sum(alpha_ij'.*im2col(B2, [3,3], 'sliding')))';
    end
    p1n_ik=p1n_ik./(repmat(sum(p1n_ik,2),1,K)+eps).*repmat(ROI,1,K);
    Pin=Pin./(repmat(sum(Pin,2),1,K)+eps).*repmat(ROI,1,K);
    %---------------------------------------------------------------% 
    
    %-----------------------Update s1_ik--------------------------------%
    s1_ik=Pi.*Pin;
    s1_ik=s1_ik./(repmat(sum(s1_ik,2),1,K)+eps);
    %----------------------------------------------------------------%
    %-----------------------Update q1_ik--------------------------------%
    q1_ik=p1_ik.*p1n_ik;
    q1_ik=q1_ik./(repmat(sum(q1_ik,2),1,K)+eps);
    %----------------------------------------------------------------%
    %--------------------Update s1n_ik,q1n_ik------------------------------------%
    s1n_ik=zeros(m,K);q1n_ik=zeros(m,K);
    for k=1:K
        s1_ik_reshape=reshape(s1_ik(:,k),[row,column]);  
        q1_ik_reshape=reshape(q1_ik(:,k),[row,column]);
        A1=[zeros(row,1),s1_ik_reshape,zeros(row,1)];
        B1=[zeros(1,column+2);A1;zeros(1,column+2)];
        A2=[zeros(row,1),q1_ik_reshape,zeros(row,1)];
        B2=[zeros(1,column+2);A2;zeros(1,column+2)];
        s1n_ik(:,k)=(sum(alpha_ij'.*im2col(B1, [3,3], 'sliding')))';
        q1n_ik(:,k)=(sum(alpha_ij'.*im2col(B2, [3,3], 'sliding')))';
    end
    s1n_ik=s1n_ik./(repmat(sum(s1n_ik,2),1,K)+eps).*repmat(ROI,1,K);
    q1n_ik=q1n_ik./(repmat(sum(q1n_ik,2),1,K)+eps).*repmat(ROI,1,K);
    %---------------------------------------------------------------%
    
    %------------------------------Update parameter--------------------------%
    Mu_old=Mu;
    Pi=(1/(1+2*alpha+eps))*((1/2)*(q1_ik+q1n_ik)+alpha*(s1_ik+s1n_ik)).*repmat(ROI,1,K);
    Mu=sum((q1_ik'+q1n_ik').*(repmat(Y,K,1)-repmat(Delta',1,m).*S1_ij),2)./(sum((q1_ik'+q1n_ik'),2)+eps);
    Mu=Mu';
    T=sum((q1_ik'+q1n_ik').*(((repmat(Y,K,1)-repmat(Mu',1,m)-repmat(Delta',1,m).*S1_ij).^2)+(S2_ij-S1_ij.^2).*repmat(Delta',1,m).*repmat(Delta',1,m)),2)./(sum((q1_ik'+q1n_ik'),2)+eps);    
    Delta=sum((q1_ik'+q1n_ik').*S1_ij.*(repmat(Y,K,1)-repmat(Mu',1,m)),2)./(sum((q1_ik'+q1n_ik').*S2_ij,2)+eps);
    Delta=Delta';
    Sigma=T+Delta'.*Delta'+0.00001;
    Lambda=(Sigma.^(-1.0/2)).*Delta'./(sqrt(1-Delta'.*(Sigma.^(-1)).*Delta')+eps);
    Lambda=Lambda';

    %----------------------------------------------------------------%
    if sqrt(sum(sum((Mu_old-Mu).^2)))<=epsilon
        break;
    end
    Mu_old=Mu;
        [~,nn]=max(Z);
      [~,wz]=sort(Mu(1,:));
      nn=reshape(nn,[row,column]);
      out2=nn;
      for i=1:K
        out2(nn==wz(i))=50*(i-1);
      end 
%       iterNums=['segmentation: ',num2str(t), ' iterations'];    
%     imshow(out2,[]),title(iterNums); colormap(gray);
%     pause(0.1)
    

end

[~,nn]=max(Z);
      [~,wz]=sort(Mu);
      nn=reshape(nn,[row,column]);
      out2=nn;
      for i=1:K
        out2(nn==wz(i))=50*(i);
      end      
imshow(out2.*reshape(ROI,[row,column]),[]);