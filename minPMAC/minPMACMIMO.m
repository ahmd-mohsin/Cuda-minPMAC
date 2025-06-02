% function [FEAS_FLAG, bu_a, info] = minPMACMIMO(H, Lxu, bu_min, w, cb)
%
% INPUTS: 
%  H(:,u,n): Ly x U x N channel matrix. Ly = number of receiver antennas, 
%            U = number of users and N = number of tones.
%
%     If the channel is real-bbd (cb=2), minPMAC realizes user data rates 
%     over the lower half of the tones (or equivalently, the positive
%     freqs), and directly corresponds to the input bu_min user-data rate vector.
%
%     By constast if cb=1 (cplx bbd), minPMAC realizes user data rates over
%     all tones, but uses the same real-bbd core optmization that doubles the
%     number of tones with an equivalent-gain set, so this program halves
%     the data rate internally and realizes that half bu_min rate on the 
%     lower tones, which is the input set for which results are reported.
%     
% Lxu:  1 x 1 or 1 x U vector containing each users' number of
%       transmit antennas. If Lxu is 1 x 1, each user has Lxu antennas.
%
% bu_min: U x 1 vector containing the target rates for all the users.
%
% w:  U x 1 vector containing the weights for each user's energy.
%     
% cb: =1 if H is complex baseband, and cb=2 if H is real baseband.
%   Outputs:
%       - FEAS_FLAG: indicator of achievability. 
%           FEAS_FLAG=1 if the target is achieved by a single ordering; 
%           FEAS_FLAG=2 if the target is achieved by time-sharing
%       - bu_a: U-by-1 vector showing achieved sum rate of each user. 
%       - info: various length output depending on FEAS_FLAG
%           --if FEAS_FLAG=1: 1 x 4 cell array containing
%               {Rxxs, Eun, bun, theta} corresponds to the single vertex
%               there are no equal-theta user sets in this case
%           --if FEAS_FLAG=2: 1-x 6 cell array, with each row representing
%               a time-shared vertex {Rxxs, Eun, bun, theta, frac}
%               there are numclus equal-theta user sets in this case.
%
%      info's row entries in detail (one row for each vertex shared
%       - Rxxs: U-by-N cell array containing Rxx(u,n)'s if Lxu is a
%           length-U vector; or Lxu-by-Lxu-by-U-by-N tensor if Lxu is a
%           scalar. If the rate target is infeasible, output 0.
%       - Eun:  U-by-N matrix showing users' transmit energy on each tone.
%           If infeasible, output 0.
%       - bun: U-by-N matrix showing users' rate on each tone. If
%           infeasible, output 0.
%       - theta: U-by-1 Lagrangian multiplier w.r.t. target rates
%       - order: produces the order from left(best) to right for vertex
%       - frac: fraction of dimensions for each vertex in time share (FF =2
%           ONLY)
%       - cluster: index (to which cluster the user belongs; 0 means no
%           cluster)
%
% Functions called are
%       startEllipse_var_Lxu
%       minPtoneMIMO
% *********************************************************************
function [FEAS_FLAG, bu_a, info] = minPMACMIMO(H, Lxu, bu_min, w, cb)
tstart=tic;
bu_min=1/(3-cb)*bu_min;
err=1e-9;
conv_tol = 1e-1;
[Ly, ~, N] = size(H);
U = length(w);
% Yun, I don't think these next 3 lines change anything? 
  if N == 1
    H = reshape(H,Ly,[],1);
  end
w = reshape(w, U, 1);
bu_min = reshape(bu_min, U, 1);

% initialize
count = 0;
bun = zeros(U,N);
Eun = zeros(U,N);
Rxxs = cell(1,N);
[A, theta] = startEllipse_var_Lxu(H, Lxu, bu_min, w,cb);

bu_min = bu_min*log(2);

  if length(Lxu) == 1
    Lxu = ones(1,U)*Lxu;
    UNIFORM_FLAG = 1;
  end

while 1
    % solve for each tone
    parfor tone = 1:N
        [~, bun(:,tone), Rxxs{tone}, Eun(:,tone)] = minPtoneMIMO(H(:,:,tone), Lxu, theta, w); 
    end
    % update ellipsoid for theta
    g = sum(bun,2) - bu_min;                   
    % stopping criteria
    tmp_err = sqrt(g'*A*g);
    if tmp_err <= err           
        break
    end
    gt = g/tmp_err;
    tmp = A*gt;
    theta = theta - 1/(U + 1)*tmp;
    A = U^2/(U^2 - 1)*(A - 2/(U + 1)*(tmp*tmp'));
    ind = find(theta < zeros(U,1));
    
    while ~isempty(ind) % This part is to make sure that theta is feasible,
        g = zeros(U,1); 
        g(ind(1)) = -1;
        tmp = A * g / sqrt(g' * A * g);
        theta = theta - 1 / (U + 1) * tmp;
        A = U^2 / (U^2 - 1) * (A - 2 / (U + 1) * (tmp * tmp'));
        ind = find(theta < zeros(U,1));
    end   
    count = count+1;
end

% if max(Lxu) == 1
%     Rxxs = Eun;
% end
bun = (3-cb)*bun /log(2);
bu_min = (3-cb)*bu_min /log(2);
bu_a=sum(bun,2);









% --------- REORDER USERS ACCORDING TO THETA -------
% This ordering will need eventually to be reversed before returing from
% this function.
[theta , Itheta] = sort(theta, 'descend');
[~, Jtheta]=sort(Itheta);
% theta = theta(Jtheta) reverses this sort later.
bu_a=bu_a(Itheta);
bun=bun(Itheta,:);
Eun=Eun(Itheta,:);
bu_min=bu_min(Itheta);
index_end = cumsum(Lxu);
index_start = [1, index_end(1:end-1)+1];
hidx = [];
for i = Itheta'
    hidx = [hidx, index_start(i):index_end(i)];
end
Lxu = Lxu(Itheta);
index_end = cumsum(Lxu);
index_start = [1, index_end(1:end-1)+1];
H=H(:,hidx,:);
for tone=1:N
    Rxxs{tone}= Rxxs{tone}(Itheta);
end

% --------------  SIMPLE CASE OF NO EQUAL-THETA USERS -------------
% This next section is implemented only if there are no equal-theta users
% and then returns results. 
if isempty(find(abs(diff(theta)) <= 1e-5*min(theta), 1))
    FEAS_FLAG = 1;
    bu_a=sum(bun,2);
    % restore the original order
    bu_a=bu_a(Jtheta);
    bu_v=bu_a';
    theta = theta(Jtheta);
    bun=bun(Jtheta,:);
    Eun=Eun(Jtheta,:);
    for tone=1:N
       Rxxs{tone}= Rxxs{tone}(Jtheta);
    end
    % make table and exit
    info = table(bu_v, Rxxs, {Eun}, {bun}, {theta}, {Itheta(end:-1:1)}); % detailed info of boundary vertices
    info.Properties.VariableNames(2:end) = {'Rxxs', 'Eun', 'bun', 'theta', 'order'};
    toc(tstart)
    return  % ARRIVE HERE AND SIMPLE CASE IS DONE.PROGRAM OVER
end
%
%---------------------TWO OR MORE EQUAL-THETA USERS ------------------
% This point of program is only reached if there are equal theta values for
% two or more users.  A convex combination of those orders different rate
% vectors will implement vertex sharing.  The info table above does not
% exist if we are in this section.  So it will be initialized and then
% vertices added to it. 

% Only need vertices corresponding to the number of equal-theta users,
% which number will be the sizeset+1 that this section generates.  Thus,
% there is no need for U! orders to be searched (which is usually a much
% larger number.

% ----------- ENUMERATION OF POSSIBLE ORDERS ---------------
% there will be setsize+1 orders to check when this section completes as
% the top setsize+1 rows of the matrix order. 
% invorder restores the original order

spots = diff([-theta', 0]) < 1e-7*norm(theta);
sizeclus = []; % size of each cluster
numclus = 0; % number of clusters (>1 equal-theta groups)
set_thetaeq = []; % index of the first entry of each cluster

i = 1;
while i <= U
    sizeset = 1;
    flagclus = 0;
    while spots(i) == 1
        sizeset = sizeset + 1;
        if ~flagclus % enter a new cluster, save the last user
            set_thetaeq = [set_thetaeq, i];
            flagclus = 1;
            numclus = numclus + 1;
        end
        i = i+1;
    end
    if sizeset > 1
        sizeclus = [sizeclus, sizeset];
    end
    i = i+1;
end

order=repmat(1:U,sum(sizeclus-1),1);
cumsize = cumsum([0,sizeclus-1]);

for jdx = 1:numclus
    Jright = eye(sizeclus(jdx));
    Jright = [Jright(:,end), Jright(:,1:end-1)];
    Jleft = Jright';
    u_range = set_thetaeq(jdx):set_thetaeq(jdx)+sizeclus(jdx)-1; % users in cluster jdx
    for i = 1:sizeclus(jdx)-1
        order(cumsize(jdx)+i, u_range) = u_range * Jleft^(i); % all permutated orders (excluding 1,2,...U in each cluster)
    end
end


% ----------- FIND FIRST VERTEX & CREATE INFO TABLE ----------
bu_v = sum(bun,2)'; % 1xU
initialbu_v = bu_v;
initialbun = bun;
firstvertices = de2bi(0:2^U-1).*initialbu_v;

% The vertices will be stored in a table that is indexed by bu_v 
initialinfo = table(bu_v, Rxxs, {Eun}, {bun}, {theta}, {U:-1:1}); % detailed info of boundary vertices
initialinfo.Properties.VariableNames(2:end) = {'Rxxs' 'Eun' 'bun' 'theta' 'order'};

% ----- For each cluster -----------
for jdx=1:numclus
    if jdx == 1
        Sinit = repmat(eye(Ly),1,1,N);
        cumrateinit = zeros(1, N);
        for n = 1:N
            for i = 1:set_thetaeq(1)-1
                Sinit(:,:,n) = Sinit(:,:,n) + H(:,index_start(i):index_end(i),n)*Rxxs{n}{i}*H(:,index_start(i):index_end(i),n)';
            end
            cumrateinit(n) = (1/cb)*real(log2(det(Sinit(:,:,n))));
        end

    else
        for n = 1:N
            for i = set_thetaeq(jdx-1):set_thetaeq(jdx-1)+sizeclus(jdx)-1
                Sinit(:,:,n) = Sinit(:,:,n) + H(:,index_start(i):index_end(i),n)*Rxxs{n}{i}*H(:,index_start(i):index_end(i),n)';
            end
            cumrateinit(n) = (1/cb)*real(log2(det(Sinit(:,:,n))));
        end
    end
    u_range = set_thetaeq(jdx):set_thetaeq(jdx)+sizeclus(jdx)-1;
    bu_v=initialbu_v;
    known_vertices = initialinfo; % detailed info of boundary vertices
    bd_vertices = bu_v(u_range);    % track critical boundary vertices
    vertices = de2bi(0:2^sizeclus(jdx)-1).*bd_vertices;
    if min(bd_vertices - bu_min(u_range)') >= -conv_tol % achievable w/o time share
        info = known_vertices;
        info.frac = 1;
        info.clusterID = jdx;
        if jdx == 1
            Big_info = info;
        else
            Big_info = [Big_info; info];
        end
        continue;
    end
    bd_V = 1;
    for idx = cumsize(jdx)+1:cumsize(jdx)+sizeclus(jdx)-1 % add more vertices in cluster jdx
        cumrate = zeros(sizeclus(jdx)+1, N);
        cumrate(1,:) = cumrateinit;
        for n = 1:N
            rel_idx = 2;
            S = Sinit(:,:,n);
            for u = u_range
                u_or = order(jdx, u);
                S = S + H(:,index_start(u_or):index_end(u_or),n)*Rxxs{n}{u_or}*H(:,index_start(u_or):index_end(u_or),n)';
                cumrate(rel_idx,n) = (1/cb)*real(log2(det(S)));
                rel_idx = rel_idx + 1;
            end
        end
        bun = initialbun;
        bun(u_range,:) = diff(cumrate);
        bun(order(idx,:),:) = bun;
        bu_v = sum(bun, 2)';

        bd_V = bd_V + 1;
        bd_vertices_extend = [bd_vertices; bu_v(u_range)];
        known_vertices = [known_vertices; {bu_v, Rxxs, {Eun}, {bun}, {theta}, {order(idx,end:-1:1)}}];
        vertices = [vertices; de2bi(0:2^sizeclus(jdx)-1).*bu_v(u_range)];
        tess = convhulln(vertices);
        vertices = vertices(unique(tess),:);
        bd_vertices = intersect(bd_vertices_extend, vertices, 'rows', 'stable');
        tess = convhulln(vertices);
        % delete inner vertices
        if size(bd_vertices, 1) < bd_V
            to_remove = setdiff(bd_vertices_extend, bd_vertices, 'rows');
            known_vertices(ismember(known_vertices.bu_v(u_range), to_remove, 'rows'),:) = [];
            bd_V = size(bd_vertices, 1);
        end

        if inhull(bu_min(u_range)', vertices, tess, conv_tol) % bu_min achievable by time-share
            FEAS_FLAG = 2;
            frac = bd_vertices'\bu_min(u_range);
            a_v = find(frac >= err); % active vertices in time-share
            info = known_vertices(a_v, :);
            info.frac = frac(a_v);
            info.clusterID = ones(length(a_v), 1)*jdx;
            bu_a(u_range) = bd_vertices(a_v,:)'*frac(a_v);
            break;
        end
    end
    
    % write big info table
    if jdx == 1
        Big_info = info;
    else
        Big_info = [Big_info; info];
    end
end

info = Big_info;
bd_V = size(info.bu_v, 1);

% ----------- RESTORE ORIGINAL ORDER TO ALL ----------------
theta = theta(Jtheta);
temptheta = reshape(cell2mat(info.theta), U, bd_V);
temptheta = temptheta(Jtheta,:);
info.theta= mat2cell(temptheta', [ones(1,bd_V)] , U);
%
tempbun=reshape(cell2mat(info.bun),U,bd_V,N);
tempbun=tempbun(Jtheta,:,:);
tempbun=permute(tempbun,[2 1 3]);
info.bun=mat2cell(tempbun,[ones(1,bd_V)], U , N);
%
tempEun=reshape(cell2mat(info.Eun),U,bd_V,N);
tempEun=tempEun(Jtheta,:,:);
tempEun=permute(tempEun,[2 1 3]);
info.Eun=mat2cell(tempEun,[ones(1,bd_V)], U , N);
%
bu_a = bu_a(Jtheta); 
info.bu_v=info.bu_v(:,Jtheta);
%
tmporder = cell2mat(info.order);
tmporder = Itheta(tmporder);
info.order = mat2cell(tmporder, ones(1,bd_V), U);

for tone=1:N
    Rxxs{tone}= Rxxs{tone}(Jtheta);
end
for u=1:U
    for n=1:N
        info.Rxxs{n}{u}=Rxxs{n}{u};
    end
end
toc(tstart)
end

