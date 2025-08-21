#pragma GCC optimize("Ofast")
#pragma GCC target("sse4.2,avx,avx2")
#include <bits/stdc++.h>
using namespace std;

// Template C++ "full" cho các dạng graph thường gặp trong competitive programming
#define faster ios::sync_with_stdio(false); cin.tie(nullptr);
using ll = long long;
using pii = pair<int,int>;
using pll = pair<ll,ll>;
const ll LINF = (ll)9e18;
const int INF = 1e9;

/*
  HƯỚNG DẪN SỬ DỤNG: copy file này vào editor, bật/ tắt các phần cần thiết.
  Mỗi phần có một hàm/struct độc lập. Tên biến chung:
    - n: số đỉnh (1..n hoặc 0..n-1 tuỳ bài)
    - m: số cạnh
    - g: adjacency list (unweighted) -> vector<vector<int>> g(n+1)
    - wg: weighted graph -> vector<vector<pair<int,w>>> wg(n+1)
*/

// ===================== Basic Graph Containers =====================

struct Edge {
    int u, v;
    ll w;
    Edge(){}
    Edge(int _u,int _v,ll _w=0):u(_u),v(_v),w(_w){}
};

// adjacency list unweighted
using Graph = vector<vector<int>>;
// adjacency list weighted
using WGraph = vector<vector<pair<int,ll>>>;

// ===================== DSU (Union-Find) =====================
struct DSU{
    int n;
    vector<int> p, r;
    DSU(int n=0){init(n);}
    void init(int _n){ n=_n; p.resize(n+1); r.assign(n+1,0); for(int i=0;i<=n;i++) p[i]=i; }
    int find(int x){ return p[x]==x?x:p[x]=find(p[x]); }
    bool unite(int a,int b){
        a=find(a); b=find(b);
        if(a==b) return false;
        if(r[a]<r[b]) swap(a,b);
        p[b]=a;
        if(r[a]==r[b]) r[a]++;
        return true;
    }
};

// ===================== BFS (unweighted shortest paths, multi-source) =====================
vector<int> bfs(const Graph &g, const vector<int> &sources){
    int n = (int)g.size()-1;
    vector<int> dist(n+1, INF);
    deque<int> dq;
    for(int s: sources){ dist[s]=0; dq.push_back(s); }
    while(!dq.empty()){
        int u = dq.front(); dq.pop_front();
        for(int v: g[u]){
            if(dist[v] > dist[u] + 1){
                dist[v] = dist[u] + 1;
                dq.push_back(v);
            }
        }
    }
    return dist;
}

// ===================== 0-1 BFS (cạnh trọng số 0 hoặc 1) =====================
vector<int> zero_one_bfs(const vector<vector<pair<int,int>>> &g, int src){
    int n = (int)g.size()-1;
    vector<int> dist(n+1, INF);
    deque<int> dq;
    dist[src]=0; dq.push_back(src);
    while(!dq.empty()){
        int u=dq.front(); dq.pop_front();
        for(auto [v,w]: g[u]){
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                if(w==0) dq.push_front(v); else dq.push_back(v);
            }
        }
    }
    return dist;
}

// ===================== Dijkstra (non-negative weights) =====================
vector<ll> dijkstra(const WGraph &g, int src){
    int n = (int)g.size()-1;
    vector<ll> dist(n+1, LINF);
    priority_queue<pll, vector<pll>, greater<pll>> pq; // (dist, node)
    dist[src]=0; pq.push({0,src});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(d!=dist[u]) continue;
        for(auto [v,w]: g[u]){
            if(dist[v] > d + w){
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// ===================== Bellman-Ford (detect negative cycles) =====================
// Returns pair(dist, hasNegativeCycle)
pair<vector<ll>, bool> bellman_ford(int n, const vector<Edge>& edges, int src){
    vector<ll> dist(n+1, LINF);
    dist[src]=0;
    for(int i=1;i<=n-1;i++){
        bool relaxed=false;
        for(auto &e: edges){
            if(dist[e.u]==LINF) continue;
            if(dist[e.v] > dist[e.u] + e.w){
                dist[e.v] = dist[e.u] + e.w;
                relaxed=true;
            }
        }
        if(!relaxed) break;
    }
    // one more iteration to check negative cycle
    bool neg=false;
    for(auto &e: edges){
        if(dist[e.u]==LINF) continue;
        if(dist[e.v] > dist[e.u] + e.w){ neg=true; break; }
    }
    return {dist, neg};
}

// ===================== SPFA (careful: worst-case slow) =====================
vector<ll> spfa(const vector<vector<pair<int,ll>>> &g, int src){
    int n=(int)g.size()-1;
    vector<ll> dist(n+1, LINF);
    vector<char> inq(n+1,0);
    queue<int> q;
    dist[src]=0; q.push(src); inq[src]=1;
    while(!q.empty()){
        int u=q.front(); q.pop(); inq[u]=0;
        for(auto [v,w]: g[u]){
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                if(!inq[v]){ inq[v]=1; q.push(v); }
            }
        }
    }
    return dist;
}

// ===================== Kruskal (MST) =====================
ll kruskal(int n, vector<Edge> &edges){
    sort(edges.begin(), edges.end(), [](const Edge &a,const Edge &b){ return a.w < b.w; });
    DSU dsu(n);
    dsu.init(n);
    ll total = 0;
    for(auto &e: edges){
        if(dsu.unite(e.u, e.v)) total += e.w;
    }
    return total;
}

// ===================== Prim (MST) =====================
ll prim(int n, const WGraph &g, int src=1){
    vector<char> vis(n+1,0);
    priority_queue<pll, vector<pll>, greater<pll>> pq; // (w, v)
    pq.push({0, src});
    ll total = 0;
    while(!pq.empty()){
        auto [w, u] = pq.top(); pq.pop();
        if(vis[u]) continue;
        vis[u]=1; total += w;
        for(auto [v,ww]: g[u]) if(!vis[v]) pq.push({ww, v});
    }
    return total;
}

// ===================== Topological Sort (Kahn) =====================
vector<int> topo_kahn(const Graph &g){
    int n = (int)g.size()-1;
    vector<int> indeg(n+1,0);
    for(int u=1;u<=n;u++) for(int v: g[u]) indeg[v]++;
    queue<int> q;
    for(int i=1;i<=n;i++) if(indeg[i]==0) q.push(i);
    vector<int> order;
    while(!q.empty()){
        int u=q.front(); q.pop(); order.push_back(u);
        for(int v: g[u]){
            indeg[v]--;
            if(indeg[v]==0) q.push(v);
        }
    }
    return order; // nếu size != n => có chu trình
}

// ===================== Tarjan SCC =====================
struct TarjanSCC{
    int n, timer=0, scc_cnt=0;
    vector<int> disc, low, st, comp;
    vector<char> inStack;
    const Graph &g;
    TarjanSCC(const Graph &g): g(g){ n = (int)g.size()-1; disc.assign(n+1, -1); low.assign(n+1,0); comp.assign(n+1,-1); inStack.assign(n+1,0); }
    void dfs(int u){
        disc[u] = low[u] = ++timer;
        st.push_back(u); inStack[u]=1;
        for(int v: g[u]){
            if(disc[v]==-1){ dfs(v); low[u] = min(low[u], low[v]); }
            else if(inStack[v]) low[u] = min(low[u], disc[v]);
        }
        if(low[u] == disc[u]){
            while(true){
                int v = st.back(); st.pop_back(); inStack[v]=0; comp[v]=scc_cnt;
                if(v==u) break;
            }
            scc_cnt++;
        }
    }
    pair<vector<int>, int> run(){
        for(int i=1;i<=n;i++) if(disc[i]==-1) dfs(i);
        return {comp, scc_cnt};
    }
};

// ===================== Hopcroft-Karp (Maximum Bipartite Matching) =====================
struct HopcroftKarp{
    int nL, nR; // left: 1..nL, right: 1..nR
    vector<vector<int>> adj; // adj[u] = list of v in [1..nR]
    vector<int> dist, pairU, pairV;
    HopcroftKarp(int _nL, int _nR): nL(_nL), nR(_nR){ adj.assign(nL+1,{}); pairU.assign(nL+1,0); pairV.assign(nR+1,0); dist.assign(nL+1,0); }
    void addEdge(int u,int v){ adj[u].push_back(v); }
    bool bfs(){
        queue<int> q;
        for(int u=1;u<=nL;u++){
            if(pairU[u]==0){ dist[u]=0; q.push(u); }
            else dist[u]=INF;
        }
        bool reachable=false;
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(int v: adj[u]){
                if(pairV[v]==0) reachable = true;
                else if(dist[pairV[v]]==INF){ dist[pairV[v]] = dist[u]+1; q.push(pairV[v]); }
            }
        }
        return reachable;
    }
    bool dfs(int u){
        for(int v: adj[u]){
            if(pairV[v]==0 || (dist[pairV[v]]==dist[u]+1 && dfs(pairV[v]))){
                pairU[u]=v; pairV[v]=u; return true;
            }
        }
        dist[u]=INF; return false;
    }
    int maxMatching(){
        int result=0;
        while(bfs()){
            for(int u=1;u<=nL;u++) if(pairU[u]==0) if(dfs(u)) result++;
        }
        return result;
    }
};

// ===================== Dinic (Max Flow) =====================
struct Dinic {
    struct Edge { int to; ll cap; int rev; };
    int N;
    vector<vector<Edge>> G;
    vector<int> level, it;
    Dinic(int n=0){ init(n); }
    void init(int n){ N=n; G.assign(N,{}); level.assign(N,0); it.assign(N,0); }
    void addEdge(int u,int v,ll cap){
        Edge a={v,cap,(int)G[v].size()};
        Edge b={u,0,(int)G[u].size()};
        G[u].push_back(a); G[v].push_back(b);
    }
    bool bfs(int s,int t){
        fill(level.begin(), level.end(), -1);
        queue<int> q; q.push(s); level[s]=0;
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(auto &e: G[u]) if(level[e.to]<0 && e.cap>0){ level[e.to]=level[u]+1; q.push(e.to); }
        }
        return level[t]>=0;
    }
    ll dfs(int u,int t,ll f){
        if(u==t) return f;
        for(int &i=it[u]; i<(int)G[u].size(); ++i){
            Edge &e = G[u][i];
            if(e.cap>0 && level[e.to]==level[u]+1){
                ll got = dfs(e.to, t, min(f, e.cap));
                if(got>0){ e.cap -= got; G[e.to][e.rev].cap += got; return got; }
            }
        }
        return 0;
    }
    ll maxflow(int s,int t){
        ll flow=0;
        while(bfs(s,t)){
            fill(it.begin(), it.end(), 0);
            while(true){ ll f = dfs(s,t,LINF); if(!f) break; flow += f; }
        }
        return flow;
    }
};

// ===================== LCA (Binary Lifting) =====================
struct LCA{
    int n, LOG;
    vector<int> depth;
    vector<vector<int>> up; // up[k][v] = 2^k-th ancestor
    LCA(int n=0){ init(n); }
    void init(int _n){
        n=_n; LOG = 1;
        while((1<<LOG) <= n) LOG++;
        depth.assign(n+1,0);
        up.assign(LOG, vector<int>(n+1, 0));
    }
    void dfs(int u,int p,const Graph &g){
        up[0][u]=p; depth[u] = (p==0?0:depth[p]+1);
        for(int v: g[u]) if(v!=p) dfs(v,u,g);
    }
    void build(int root, const Graph &g){
        dfs(root, 0, g);
        for(int k=1;k<LOG;k++) for(int v=1;v<=n;v++) up[k][v] = up[k-1][ up[k-1][v] ];
    }
    int lca(int a,int b){
        if(depth[a] < depth[b]) swap(a,b);
        int diff = depth[a]-depth[b];
        for(int k=0;k<LOG;k++) if(diff & (1<<k)) a = up[k][a];
        if(a==b) return a;
        for(int k=LOG-1;k>=0;k--) if(up[k][a] != up[k][b]){ a = up[k][a]; b = up[k][b]; }
        return up[0][a];
    }
};

// ===================== Heavy-Light Decomposition (skeleton) =====================
struct HLD{
    int n, timer=0;
    vector<int> parent, depth, heavy, head, pos, size;
    HLD(int n=0){ init(n); }
    void init(int _n){
        n=_n; parent.assign(n+1,0); depth.assign(n+1,0); heavy.assign(n+1,-1);
        head.assign(n+1,0); pos.assign(n+1,0); size.assign(n+1,0); timer=0;
    }
    int dfs(int u,int p,const Graph &g){
        parent[u]=p; size[u]=1; int maxsz=0;
        for(int v: g[u]) if(v!=p){ depth[v]=depth[u]+1; int s = dfs(v,u,g); size[u]+=s; if(s>maxsz){ maxsz=s; heavy[u]=v; } }
        return size[u];
    }
    void decompose(int u,int h,const Graph &g){
        head[u]=h; pos[u]=++timer;
        if(heavy[u]!=-1) decompose(heavy[u], h, g);
        for(int v: g[u]) if(v!=parent[u] && v!=heavy[u]) decompose(v, v, g);
    }
    void build(int root, const Graph &g){ dfs(root,0,g); decompose(root, root, g); }
    // Now pos[u] gives position for segtree/bit usage. Implement segment tree separately.
};

// ===================== Tree DP skeleton =====================
// Example: compute subtree sizes, or DP on tree with parent
void tree_dp_example(int u,int p,const Graph &g, vector<int> &subsz){
    subsz[u]=1;
    for(int v: g[u]) if(v!=p){
        tree_dp_example(v,u,g,subsz);
        subsz[u]+=subsz[v];
    }
}

// ===================== Notes =====================
// - Các template có thể điều chỉnh chỉ số đỉnh (0-based / 1-based) theo bài.
// - Kiểm tra giới hạn n,m để chọn thuật toán phù hợp.
// - Một số hàm (Hopcroft-Karp) giả sử các đỉnh trái/phải bắt đầu từ 1.
// - Thêm xử lí IO, đọc đồ thị, gọi các hàm tuỳ bài.

// ===================== Minimal main example (commented) =====================
/*
int main(){
    faster
    int n,m; cin>>n>>m;
    Graph g(n+1);
    for(int i=0;i<m;i++){ int u,v; cin>>u>>v; g[u].push_back(v); g[v].push_back(u); }
    // BFS example
    auto d = bfs(g, {1});
    for(int i=1;i<=n;i++) cout << (d[i]==INF?-1:d[i]) << '\n';
    return 0;
}
*/

// ===================== Additional Graph Algorithms & Ghi chú (Tiếng Việt) =====================
// Phần dưới đây bổ sung: A*, Floyd-Warshall, Johnson (APSP sparse), Kosaraju, articulation/bridges,
// Euler path (Hierholzer), Edmonds-Karp, Push-Relabel, Chu-Liu/Edmonds (directed MST), Centroid Decomposition.
// Các ghi chú đều bằng tiếng Việt, kèm code ví dụ. Chỉnh lại chỉ số (0-based/1-based) theo bài thực tế.

// --------------------- A* (shortest path with heuristic) ---------------------
// Sử dụng khi muốn tìm đường ngắn nhất và có heuristic (ước lượng) hợp lệ (admissible).
// heuristic(u) phải luôn <= thật sự cost từ u tới goal.
vector<ll> astar(const WGraph &g, int src, int target, function<ll(int)> heuristic){
    int n = (int)g.size()-1;
    vector<ll> gscore(n+1, LINF), fscore(n+1, LINF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq; // (f, node)
    gscore[src]=0; fscore[src]=heuristic(src);
    pq.push({fscore[src], src});
    while(!pq.empty()){
        auto [f,u] = pq.top(); pq.pop();
        if(u==target) return gscore; // trả về gscore (có thể dừng sớm)
        if(f != fscore[u]) continue;
        for(auto [v,w]: g[u]){
            ll tentative = gscore[u] + w;
            if(tentative < gscore[v]){
                gscore[v]=tentative;
                fscore[v]=tentative + heuristic(v);
                pq.push({fscore[v], v});
            }
        }
    }
    return gscore;
}

// --------------------- Floyd-Warshall (APSP dense) ---------------------
// Dùng khi n nhỏ (<= 400..500)
vector<vector<ll>> floyd_warshall(int n, const vector<Edge> &edges){
    vector<vector<ll>> dist(n+1, vector<ll>(n+1, LINF));
    for(int i=1;i<=n;i++) dist[i][i]=0;
    for(auto &e: edges) dist[e.u][e.v] = min(dist[e.u][e.v], e.w);
    for(int k=1;k<=n;k++) for(int i=1;i<=n;i++) if(dist[i][k]<LINF) for(int j=1;j<=n;j++) if(dist[k][j]<LINF) dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    return dist;
}

// --------------------- Johnson (APSP for sparse graphs) ---------------------
// Dùng khi có trọng số âm nhưng không có chu trình âm. Ý tưởng: thêm node 0 nối cạnh 0->i weight 0, chạy BF
// để tìm potential h[]; tái định giá w'(u,v)=w(u,v)+h[u]-h[v] >= 0; rồi chạy Dijkstra từ mỗi đỉnh.
vector<vector<ll>> johnson(int n, vector<Edge> edges){
    // thêm node 0
    vector<Edge> ext = edges;
    for(int i=1;i<=n;i++) ext.emplace_back(0,i,0);
    auto [h, neg] = bellman_ford(n, ext, 0);
    if(neg) return {}; // có chu trình âm -> không dùng được
    // tái định giá
    WGraph g(n+1);
    for(auto &e: edges){
        ll w2 = e.w + h[e.u] - h[e.v];
        g[e.u].push_back({e.v, w2});
    }
    vector<vector<ll>> all(n+1);
    for(int s=1;s<=n;s++){
        auto d = dijkstra(g, s);
        // chỉnh lại dist về giá trị ban đầu
        for(int i=1;i<=n;i++) if(d[i]<LINF) d[i] = d[i] - h[s] + h[i];
        all[s]=d;
    }
    return all;
}

// --------------------- Kosaraju SCC (alternative) ---------------------
vector<int> kosaraju_scc(const Graph &g){
    int n=(int)g.size()-1;
    vector<int> order; order.reserve(n);
    vector<char> vis(n+1,0);
    function<void(int)> dfs1 = [&](int u){ vis[u]=1; for(int v: g[u]) if(!vis[v]) dfs1(v); order.push_back(u); };
    for(int i=1;i<=n;i++) if(!vis[i]) dfs1(i);
    // build reverse graph
    Graph rg(n+1);
    for(int u=1;u<=n;u++) for(int v: g[u]) rg[v].push_back(u);
    vector<int> comp(n+1,-1);
    int cid=0;
    function<void(int)> dfs2 = [&](int u){ comp[u]=cid; for(int v: rg[u]) if(comp[v]==-1) dfs2(v); };
    for(int i=n-1;i>=0;i--){ int u=order[i]; if(comp[u]==-1){ dfs2(u); cid++; } }
    return comp;
}

// --------------------- Articulation Points & Bridges ---------------------
struct CutBridge{
    int n, timer=0;
    vector<int> tin, low;
    vector<char> isCut;
    vector<pair<int,int>> bridges;
    CutBridge(int n=0){ init(n); }
    void init(int _n){ n=_n; tin.assign(n+1,-1); low.assign(n+1,-1); isCut.assign(n+1,0); timer=0; bridges.clear(); }
    void dfs(int u,int p,const Graph &g){
        tin[u]=low[u]=++timer; int children=0;
        for(int v: g[u]){
            if(v==p) continue;
            if(tin[v]!=-1){ low[u]=min(low[u], tin[v]); }
            else{
                dfs(v,u,g);
                low[u]=min(low[u], low[v]);
                if(low[v] >= tin[u]) children++;
                if(low[v] > tin[u]) bridges.emplace_back(u,v);
            }
        }
        if(p==0 && children>1) isCut[u]=1;
        if(p!=0 && children>0){ /* handled above by low[v] >= tin[u]*/ }
    }
    pair<vector<char>, vector<pair<int,int>>> run(const Graph &g){
        init((int)g.size()-1);
        for(int i=1;i<=n;i++) if(tin[i]==-1) dfs(i,0,g);
        return {isCut, bridges};
    }
};

// --------------------- Eulerian Path / Circuit (Hierholzer) ---------------------
// Giải thuật Hierholzer: O(E). Hỗ trợ hướng/không hướng.
vector<int> eulerian_path_undirected(int n, vector<pair<int,int>> edges){
    // edges: list of (u,v) 1-based. Nếu multi-edge, ta cần id để đánh dấu.
    vector<vector<pair<int,int>>> adj(n+1);
    int m = edges.size();
    for(int i=0;i<m;i++){
        auto [u,v] = edges[i];
        adj[u].push_back({v,i}); adj[v].push_back({u,i});
    }
    vector<char> used(m,0);
    vector<int> st, path;
    int start = 1;
    for(int i=1;i<=n;i++) if(!adj[i].empty()){ start=i; break; }
    st.push_back(start);
    vector<int> idx(n+1,0);
    while(!st.empty()){
        int v = st.back();
        while(idx[v] < (int)adj[v].size() && used[adj[v][idx[v]].second]) idx[v]++;
        if(idx[v] == (int)adj[v].size()){ path.push_back(v); st.pop_back(); }
        else{
            auto [to,id] = adj[v][idx[v]++]; used[id]=1; st.push_back(to);
        }
    }
    reverse(path.begin(), path.end());
    return path; // nếu path size != m+1 thì không tồn tại
}

// --------------------- Edmonds-Karp (Max Flow) ---------------------
ll edmonds_karp(int N, vector<vector<pair<int,ll>>> g_adj, int s, int t){
    // g_adj: adjacency with capacities; we will build residual graph indices
    // Simpler: build Dinic or reuse Dinic. Đây chỉ là ví dụ khái niệm.
    // Recommend dùng Dinic trong contest vì nhanh hơn.
    return Dinic(N).maxflow(s,t);
}

// --------------------- Push-Relabel (simplified) ---------------------
struct PushRelabel{
    struct Edge{int to; ll cap; int rev;};
    int N; vector<vector<Edge>> G; vector<ll> excess; vector<int> height, active; queue<int> q;
    PushRelabel(int n=0){ init(n); }
    void init(int n){ N=n; G.assign(N,{}); excess.assign(N,0); height.assign(N,0); active.assign(N,0); }
    void addEdge(int u,int v,ll cap){ G[u].push_back({v,cap,(int)G[v].size()}); G[v].push_back({u,0,(int)G[u].size()-1}); }
    void push(Edge &e, int u){ ll send = min(excess[u], e.cap); if(send==0) return; e.cap -= send; G[e.to][e.rev].cap += send; excess[e.to] += send; excess[u]-=send; if(!active[e.to]){ active[e.to]=1; q.push(e.to);} }
    void relabel(int u){ int d = INT_MAX; for(auto &e: G[u]) if(e.cap>0) d = min(d, height[e.to]); if(d<INT_MAX) height[u]=d+1; }
    ll maxflow(int s,int t){ height[s]=N; active[s]=active[t]=1; for(auto &e: G[s]){ excess[s]+=0; push(e,s);} // initialize preflow using pushes
        for(auto &e: G[s]){ ll c = e.cap; if(c>0) push(e, s); }
        while(!q.empty()){
            int u=q.front(); q.pop(); active[u]=0;
            for(auto &e: G[u]) if(excess[u]>0) if(e.cap>0 && height[u]==height[e.to]+1) push(e,u);
            if(excess[u]>0){ relabel(u); if(!active[u]){ active[u]=1; q.push(u); } }
        }
        ll flow = 0; for(auto &e: G[s]) flow += G[e.to][e.rev].cap; return flow;
    }
};

// --------------------- Chu-Liu/Edmonds (Directed MST) ---------------------
// Tìm cây khung nhỏ nhất của đồ thị có hướng (arborescence) từ root.
ll directed_mst(int n, int root, const vector<Edge> &edges_in){
    // returns total weight or -1 if not possible
    const ll INFLL = LINF;
    ll res = 0;
    int N = n;
    vector<Edge> edges = edges_in;
    while(true){
        vector<ll> in(N+1, INFLL);
        vector<int> pre(N+1, -1);
        for(auto &e: edges) if(e.u!=e.v && e.w < in[e.v]){ in[e.v]=e.w; pre[e.v]=e.u; }
        in[root]=0;
        for(int i=1;i<=N;i++) if(in[i]==INFLL) return -1; // some node unreachable
        int cnt=0; vector<int> id(N+1, -1), vis(N+1, -1);
        for(int i=1;i<=N;i++) res += in[i];
        for(int i=1;i<=N;i++){
            int v=i;
            while(vis[v]!=i && id[v]==-1 && v!=root){ vis[v]=i; v=pre[v]; }
            if(v!=root && id[v]==-1){ // found cycle
                cnt++; for(int u=pre[v]; u!=v; u=pre[u]) id[u]=cnt; id[v]=cnt; }
        }
        if(cnt==0) break; // no cycle
        for(int i=1;i<=N;i++) if(id[i]==-1) id[i]=++cnt;
        // re-label nodes
        vector<Edge> newEdges;
        for(auto &e: edges){
            int u = id[e.u], v = id[e.v]; ll w = e.w;
            if(u!=v) newEdges.emplace_back(u, v, w - in[e.v]);
        }
        N = cnt; root = id[root]; edges = move(newEdges);
    }
    return res;
}

// --------------------- Centroid Decomposition (tree) ---------------------
struct CentroidDecomposition{
    int n; Graph g; vector<char> removed; vector<int> subsz;
    CentroidDecomposition(int n=0){ init(n); }
    void init(int _n){ n=_n; g.assign(n+1,{}); removed.assign(n+1,0); subsz.assign(n+1,0); }
    int dfs_sz(int u,int p){ subsz[u]=1; for(int v: g[u]) if(v!=p && !removed[v]) subsz[u]+=dfs_sz(v,u); return subsz[u]; }
    int find_centroid(int u,int p,int sz){
        for(int v: g[u]) if(v!=p && !removed[v]) if(subsz[v] > sz/2) return find_centroid(v,u,sz);
        return u;
    }
    void build(int u,int p){
        int sz = dfs_sz(u, -1);
        int c = find_centroid(u, -1, sz);
        removed[c]=1;
        // xử lý centroid c (vd: build distances, answer queries...) -> tuỳ bài
        for(int v: g[c]) if(!removed[v]) build(v, c);
    }
};

// --------------------- Ghi chú tóm tắt---------------------
/*
GHI CHÚ CHUNG:
- Luôn kiểm tra mô tả bài: đồ thị có hướng hay vô hướng? trọng số âm? đa cạnh? đa thành phần?
- Chọn thuật toán theo giới hạn:
  * n <= ~400: Floyd-Warshall có thể làm APSP.
  * n tới 1e5, m tới 2e5: dùng Dijkstra (O((n+m) log n)), BFS, DSU, Kahn, Tarjan/Kosaraju, Dinic.
  * Cạnh có trọng số 0/1: 0-1 BFS.
  * Có cạnh âm nhưng không chu trình âm: Bellman-Ford hoặc Johnson.
  * Muốn maxflow: Dinic thường đủ; với dense/cực lớn có thể dùng Push-Relabel.
  * Muốn matching trên bipartite: Hopcroft-Karp. Nếu non-bipartite/weighted -> blossom algorithm (không được đưa ở đây).
- Dùng adjacency list cho đa phần bài contest. Tránh adjacency matrix nếu m lớn.
- Với state-space implicit (ví dụ: vị trí + trạng thái), dựng đồ thị ẩn và chạy BFS/Dijkstra trên state.
- Với cây (tree): ghi nhớ LCA, HLD, centroid decomposition, tree DP. Cây có thể giảm bớt độ phức tạp so với đồ thị chung.

MẸO: Giữ bộ snippets sẵn để copy-paste vào contest. Tối giản IO, và chú thích rõ ràng (1-based/0-based).
*/
