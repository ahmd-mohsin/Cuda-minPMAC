% function [FEAS_FLAG, bu_a, info] = minPMACMIMO_reformulated(H, Lx, bu_min, w, cb)
% % minPMAC_MIMO  Minimum‑power MAC for heterogeneous‑antenna MIMO users
% % -------------------------------------------------------------------
% %   H  : Ly × (sum(Lx)) × N tensor (users concatenated horizontally)
% %   Lx : 1×U vector – #TX antennas per user
% %   bu_min : 1×U vector – required aggregate rates (bits)
% %   w  : 1×U vector – per‑user energy weights
% %   cb : 1 or 2 – log base factor (1→log2 already, 2→ln)
% % -------------------------------------------------------------------
% % Returns
% %   FEAS_FLAG : 0 infeasible | 1 feasible (single order) | 2 feasible (time‑sharing)
% %   bu_a      : 1×U achieved rates
% %   info      : struct with detailed solution data
% % -------------------------------------------------------------------
% 
%     tic;
%     [Ly, Ltot, N] = size(H);
%     U = numel(Lx);
% 
%     % antenna index ranges for each user
%     idx_end   = cumsum(Lx);
%     idx_start = [1, idx_end(1:end-1)+1];
% 
%     %% 1) Scale rates for convex program
%     bu_req = cb * bu_min(:);   % column
% 
%     %% 2) CVX optimisation – single block‑diagonal covariance per tone
%     subsets = powerSet(1:U); subsets = subsets(2:end); % drop empty set
% 
%     disp('CVX: solving min‑power MAC (MIMO)');
%     cvx_begin quiet
%         variable b(U,N) nonnegative;
%         % one big covariance per tone (Hermitian PSD)
%         variable Rxx(Ltot, Ltot, N) hermitian semidefinite;
%         dual variable theta;
% 
%         %% rate constraints
%         theta : sum(b,2) >= bu_req;   % aggregate per‑user
% 
%         %% subset MAC capacity constraints for each tone
%         for s = 1:length(subsets)
%             subset = subsets{s};
%             for n = 1:N
%                 Q = zeros(Ly,Ly);
%                 for u = subset
%                     blk = idx_start(u):idx_end(u);
%                     Hu  = H(:, blk, n);
%                     Q   = Q + Hu * Rxx(blk, blk, n) * Hu';
%                 end
%                 cap = log_det( eye(Ly) + Q )/log(2);
%                 sum( b(subset,n) ) <= cap;
%             end
%         end
% 
%         %% objective: weighted sum energy
%         obj = 0;
%         for u = 1:U
%             blk = idx_start(u):idx_end(u);
%             for n = 1:N
%                 obj = obj + w(u) * trace( Rxx(blk, blk, n) );
%             end
%         end
%         minimize(obj);
%     cvx_end
%     disp(['CVX status: ' cvx_status]);
% 
%     if ~strcmp(cvx_status,'Solved') && ~strcmp(cvx_status,'Inaccurate/Solved')
%         FEAS_FLAG = 0; bu_a = nan(1,U); info = struct(); return; end
% 
%     %% 3) post‑processing
%     % average transmit energies
%     Eu_a = zeros(1,U);
%     for u = 1:U
%         blk = idx_start(u):idx_end(u);
%         for n = 1:N
%             Eu_a(u) = Eu_a(u) + trace( Rxx(blk, blk, n) );
%         end
%     end
%     Eu_a = Eu_a / N;
% 
%     % rescale rates back
%     b      = b  / cb;
%     bu_req = bu_req / cb;
% 
%     %% 4) decode‑order clustering by Lagrange multipliers
%     [clusters,~,theta_vals] = identify_clusters(theta);
%     orders = generate_all_orders(clusters);
% 
%     if size(orders,1) == 1
%         % single order – compute achievable rates
%         [~, bu_a, ~] = compute_rates_given_order(H,Rxx,orders(1,:),Ly,U,N,cb,Lx,idx_start,idx_end);
%         FEAS_FLAG = 1;
%         weights   = 1;
%         orderings = orders;
%     else
%         [weights, orderings, bu_mat] = time_share_weights(H,Rxx,orders,bu_min,w,Ly,U,N,cb,Lx,idx_start,idx_end);
%         FEAS_FLAG = any(weights>0) * (1 + (nnz(weights)>1));
%         bu_a = (bu_mat*weights)';
%     end
% 
%     %% 5) info struct
%     info = struct('H',H,'Lx',Lx,'w',w,'cb',cb,'idx_start',idx_start,'idx_end',idx_end,...
%                   'clusters',clusters,'orderings',{orderings},'time_sharing_weights',weights,...
%                   'Eu_a',Eu_a,'bu_a',bu_a,'theta',theta_vals,'FEAS_FLAG',FEAS_FLAG);
%     disp(['Total elapsed: ' num2str(toc,3) ' s']);
% end
% 
% %% ----------------------------------------------------------------------
% %% helper utilities
% function pSet = powerSet(S)
%     n = numel(S); pSet = cell(1,2^n);
%     for i = 0:2^n-1
%         pSet{i+1} = S(logical(bitget(i,n:-1:1)));
%     end
% end
% 
% function [clusters, uniq, theta_val] = identify_clusters(theta)
%     theta_val = full(theta(:)); tol = 1e-12;
%     [uniq,~,grp] = uniquetol(theta_val,tol);  % groups by tolerance
%     clusters = cell(numel(uniq),1);
%     for k=1:numel(uniq), clusters{k}=find(grp==k); end
% end
% 
% function orders = generate_all_orders(clusters)
%     orders = perms(clusters{1});
%     for c = 2:numel(clusters)
%         pc = perms(clusters{c}); tmp = [];
%         for i=1:size(orders,1)
%             for j=1:size(pc,1)
%                 tmp = [tmp; orders(i,:) pc(j,:)]; %#ok<AGROW>
%             end
%         end
%         orders = tmp;
%     end
% end
% 
% function [b_mat, bu_vec, Eun] = compute_rates_given_order(H,Rxx,order,...
%         Ly,U,N,cb,Lx,idx_start,idx_end)
%     % per‑tone successive decoding according to "order" (1×U)
%     b_mat = zeros(U,N);
%     for n = 1:N
%         for k = 1:U
%             u = order(k);
%             % cumulative covariances
%             Sig  = zeros(Ly);
%             for j = k:U
%                 uj = order(j);
%                 blk = idx_start(uj):idx_end(uj);
%                 Sig = Sig + H(:,blk,n)*Rxx(blk,blk,n)*H(:,blk,n)';
%             end
%             Intf = zeros(Ly);
%             for j = k+1:U
%                 uj = order(j);
%                 blk = idx_start(uj):idx_end(uj);
%                 Intf = Intf + H(:,blk,n)*Rxx(blk,blk,n)*H(:,blk,n)';
%             end
%             b_mat(u,n) = (log2(det(eye(Ly)+Sig))-log2(det(eye(Ly)+Intf)))/cb;
%         end
%     end
%     bu_vec = sum(b_mat,2);
%     Eun = [];% optional
% end
% 
% function [wts, orderings, bu_mat] = time_share_weights(H,Rxx,orders,bu_min,w,...
%         Ly,U,N,cb,Lx,idx_start,idx_end)
%     K = size(orders,1);
%     bu_mat = zeros(U,K);
%     orderings = cell(K,1);
%     for k = 1:K
%         orderings{k} = orders(k,:);
%         [~, bu_k, ~] = compute_rates_given_order(H,Rxx,orders(k,:),Ly,U,N,cb,Lx,idx_start,idx_end);
%         bu_mat(:,k) = bu_k;
%     end
%     tol = 1e-4;
%     cvx_begin quiet
%         variable wts(K) nonnegative;
%         variable z(K) binary;
%         minimize( sum(z) );
%         sum(wts) == 1;
%         abs( bu_mat*wts - bu_min(:) ) <= tol;
%         wts <= z;
%     cvx_end
% end

function [FEAS_FLAG, bu_a, info] = minPMACMIMO_reformulated(H, Lx, bu_min, w, cb)
% minPMAC_MIMO  – minimum‑power multi‑user MIMO MAC with heterogeneous Tx antennas
% -------------------------------------------------------------------------
% INPUTS
%   H       : Ly × (sum(Lx)) × N array, users concatenated across columns per tone
%   Lx      : 1×U vector, antennas per user (sum(Lx)=columns of H)
%   bu_min  : 1×U target aggregate rates (bits)
%   w       : 1×U positive weights for energy objective
%   cb      : log‑base factor (1 ⇒ already log2, 2 ⇒ natural‑log → divide by ln2)
% OUTPUTS
%   FEAS_FLAG : 0 infeasible | 1 feasible single order | 2 feasible time‑sharing
%   bu_a      : 1×U achieved aggregate rates (bits)
%   info      : struct with solver details and energies
% -------------------------------------------------------------------------
% 2025‑04‑25  S.B.

    tic;
    [Ly, Ltot, N] = size(H);
    U = numel(Lx);

    % antenna index map per user
    idx_end   = cumsum(Lx);
    idx_start = [1, idx_end(1:end-1)+1];

    %% ----- 1   scale rate targets for convex formulation -----
    bu_req = cb * bu_min(:);  % column vector

    %% ----- 2   CVX optimisation -----
    subsets = powerSet(1:U); subsets = subsets(2:end); % remove Ø
    disp('CVX: solving min‑power MAC (MIMO) …');
    cvx_begin quiet
        variable b(U,N) nonnegative;
        % One covariance per tone (block‑diagonal slices)
        variable Rxx(Ltot, Ltot, N) hermitian semidefinite;
        dual variable theta;

        % per‑user aggregate‑rate constraints
        theta : sum(b,2) >= bu_req;

        % MAC polymatroid subset constraints
        for s = 1:length(subsets)
            S = subsets{s};
            for n = 1:N
                Q = zeros(Ly);
                for u = S
                    blk = idx_start(u):idx_end(u);
                    Hu  = H(:, blk, n);
                    Q   = Q + Hu * Rxx(blk, blk, n) * Hu';
                end
                cap = log_det( eye(Ly) + Q ) / log(2);
                sum( b(S,n) ) <= cap;
            end
        end

        % objective – weighted sum transmit energy
        obj = 0;
        for u = 1:U
            blk = idx_start(u):idx_end(u);
            for n = 1:N
                obj = obj + w(u) * trace( Rxx(blk, blk, n) );
            end
        end
        minimize(obj);
    cvx_end
    disp(['CVX status: ' cvx_status]);

    if ~strcmp(cvx_status,'Solved') && ~strcmp(cvx_status,'Inaccurate/Solved')
        FEAS_FLAG = 0; bu_a = nan(1,U); info = struct(); return; end

    %% ----- 3   convert Rxx to full numeric 3‑D array (needed for indexing) -----
    Rxx = full(Rxx);                     % CVX may return sparse pages
    if ndims(Rxx)==2 && N>1              % flattened → reshape
        Rxx = reshape(Rxx, Ltot, Ltot, N);
    end

    %% ----- 4   average transmit energies -----
    Eu_a = zeros(1,U);
    for u = 1:U
        blk = idx_start(u):idx_end(u);
        for n = 1:N
            Eu_a(u) = Eu_a(u) + trace( Rxx(blk, blk, n) );
        end
    end
    Eu_a = Eu_a / N;

    %% ----- 5   rescale rates back -----
    b      = b  / cb;
    bu_req = bu_req / cb;

    %% ----- 6   Lagrange‑multiplier clustering & decoding orders -----
    [clusters, ~, theta_val] = identify_clusters(theta);
    orders = generate_all_orders(clusters);

    if size(orders,1) == 1
        % Single decoding order – compute achievable rates
        [~, bu_a] = compute_rates_given_order(H,Rxx,orders(1,:),Ly,U,N,cb,idx_start,idx_end);
        FEAS_FLAG  = 1;
        weights    = 1;                 % dummy
        orderings  = orders;
    else
        % Multiple orders – time‑sharing optimisation
        [weights, orderings, bu_mat] = time_share_weights(H,Rxx,orders,bu_min,Ly,U,N,cb,idx_start,idx_end);
        FEAS_FLAG = any(weights>0) * (1 + (nnz(weights)>1));
        bu_a = (bu_mat * weights)';
    end

    %% ----- 7   info struct -----
    info = struct('H',H,'Lx',Lx,'w',w,'cb',cb,'idx_start',idx_start,'idx_end',idx_end,...
                  'clusters',{clusters},'orderings',{orderings},'weights',weights,...
                  'Eu_a',Eu_a,'bu_a',bu_a,'theta',theta_val,'FEAS_FLAG',FEAS_FLAG);
    disp(['Done. Total elapsed: ' num2str(toc,3) ' s']);
end

%% =====================================================================
%% helper: powerset (cell array)
function pSet = powerSet(S)
    n = numel(S); pSet = cell(1,2^n);
    for i = 0:2^n-1
        pSet{i+1} = S(logical(bitget(i,n:-1:1)));
    end
end

%% helper: group users by equal multipliers
function [clusters, uniq, theta_vec] = identify_clusters(theta_dual)
    theta_vec = full(theta_dual(:));
    tol  = 1e-12;
    [uniq, ~, g] = uniquetol(theta_vec, tol);
    clusters = cell(numel(uniq),1);
    for k = 1:numel(uniq), clusters{k} = find(g==k); end
end

%% helper: generate all decoding orders consistent with clusters
function orders = generate_all_orders(clusters)
    orders = perms(clusters{1});
    for c = 2:numel(clusters)
        pc = perms(clusters{c}); tmp = [];
        for i = 1:size(orders,1)
            for j = 1:size(pc,1)
                tmp(end+1,:) = [orders(i,:) pc(j,:)]; %#ok<AGROW>
            end
        end
        orders = tmp;
    end
end

%% helper: compute tone‑wise & aggregate rates for a fixed order
function [b_mat, bu_vec] = compute_rates_given_order(H,Rxx,order,Ly,U,N,cb,idx_start,idx_end)
    b_mat = zeros(U,N);
    for n = 1:N
        Rn = Rxx(:,:,n);
        for k = 1:U
            u = order(k);
            Sig = zeros(Ly);
            for j = k:U
                uj  = order(j);
                blk = idx_start(uj):idx_end(uj);
                Hj  = H(:, blk, n);
                Sig = Sig + Hj * Rn(blk,blk) * Hj';
            end
            Intf = zeros(Ly);
            for j = k+1:U
                uj  = order(j);
                blk = idx_start(uj):idx_end(uj);
                Hj  = H(:, blk, n);
                Intf = Intf + Hj * Rn(blk,blk) * Hj';
            end
            b_mat(u,n) = (log2(det(eye(Ly)+Sig)) - log2(det(eye(Ly)+Intf))) / cb;
        end
    end
    bu_vec = sum(b_mat,2);
end

%% helper: mixed‑integer LP for time‑sharing weights
function [wts, orderings, bu_mat] = time_share_weights(H,Rxx,orders,bu_min,Ly,U,N,cb,idx_start,idx_end)
    K = size(orders,1);
    bu_mat = zeros(U,K);
    orderings = cell(K,1);
    for k = 1:K
        orderings{k} = orders(k,:);
        [~, bu_k] = compute_rates_given_order(H,Rxx,orders(k,:),Ly,U,N,cb,idx_start,idx_end);
        bu_mat(:,k) = bu_k;
    end
    tol = 1e-4;
    cvx_begin quiet
        variable wts(K) nonnegative;
        variable z(K) binary;        % sparsity indicator
        minimize( sum(z) );
        sum(wts) == 1;
        abs( bu_mat*wts - bu_min(:) ) <= tol;
        wts <= z;
    cvx_end
end

