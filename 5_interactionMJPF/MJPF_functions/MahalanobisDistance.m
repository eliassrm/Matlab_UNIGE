function d=MahalanobisDistance(A, B, covA, covB)
% Return mahalanobis distance of two data matrices 
% A and B (row = object, column = feature)
% @author: Kardi Teknomo
% http://people.revoledu.com/kardi/index.html
% % [n1, k1]=size(A);
% % [n2, k2]=size(B);
% % n=n1+n2;
% % if(k1~=k2)
% %     disp('number of columns of A and B must be the same')
% % else
% %     xDiff=mean(A)-mean(B);       % mean difference row vector
% %     cA=Covariance(A);
% %     cB=Covariance(B);
% %     pC=n1/n*cA+n2/n*cB;          % pooled covariance matrix
% %     d=sqrt(xDiff*inv(pC)*xDiff'); % mahalanobis distance
    
    %% akr:
    % A is the mean of the first set (mean of x)
    % B is the mean of the second set (mean of S)
    % covA is the covariance of A (cov of x)
    % covB is the covariance of B (cov of S)
    n1 = 36;
    n2 = 36;
    n = n1 + n2;
    xDiff=A-B;       % mean difference row vector
%     cA=Covariance(A);
%     cB=Covariance(B);
% % %     pC=n1/n*covA+n2/n*covB;          % pooled covariance matrix
% % %     d=sqrt(xDiff*inv(pC)*xDiff'); % mahalanobis distance
%% I commented the previous two lines since maha dist calculate the distance from a point to a distribution set (taking mean and cov)
    d=sqrt(xDiff*inv(covB)*xDiff'); % mahalanobis distance
%     1)
% %     d=sqrt(xDiff*(covB^(-1))*xDiff');
% %     d=sqrt(xDiff*qr(pC)*xDiff');
% % 2)
% % d=sqrt(xDiff*inv(covB)*xDiff'); 
end