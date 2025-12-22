# STL

## list(双向链表)

```c++
//定义一个list 
    list<int>node; 
//为链表赋值，例如定义一个包含n个结点的链表
    for(int i = 1; i <= n; i ++){
        node.push_back(i);
    } 
//遍历链表，用it遍历链表，例如从头到尾遍历
    list<int>::iterator it= node.begin();
    while (node.size() > 1){    //list的大小由STL管理
        it++; 
        if(it == node.end()){   //循环链表，end()是1ist末端下一位置
            it = node.begin();
            break;
        }
    }
//删除一个结点
    list<int>::iterator next = ++it;
    if(next == node.end()) next = node.begin();  //循环链表
    node.erase(--it);   //删除这个结点，并将it指向前一个节点，避免再次访问it时发生错误
    it = next;   //更新it 
```

---

# [数论](数论.md 数论)


---



---

# 动态规划

## 区间dp

> ***石子合并问题***  
**题目描述**
设有 N(N≤300) 堆石子排成一排，其编号为 1,2,3,⋯,N。每堆石子有一定的质量mi(mi≤1000)。现在要将这 N 堆石子合并成为一堆。每次只能合并相邻的两堆，合并的代价为这两堆石子的质量之和，合并后与这两堆石子相邻的石子将和新堆相邻。合并时由于选择的顺序不同，合并的总代价也不相同。试找出一种合理的方法，使总的代价最小，并输出最小代价。  
**输入格式**
第一行，一个整数 N。
第二行，N 个整数 mi。
**输出格式**
输出仅一个整数，也就是最小代价。
**样例**
*input*
4
2 5 3 1  
*output*
22

```c++
void solve(void){
    int n; cin >> n;
    vector<int> arr(n + 1, 0);
    vector<int> pre(n + 1, 0);
    for(int i = 1; i <= n; i++){
        cin >> arr[i];
        pre[i] = pre[i - 1] + arr[i];
    } 
    vector<vector<int> > dp(n + 1, vector<int>(n + 1, INT_MAX));
    for(int i = 1; i <= n; i ++){
        dp[i][i] = 0;
    }
    for(int len = 2; len <= n; len ++){ // 枚举长度
        for(int i = 1; i + len - 1 <= n; i ++){ // 起始位置
            int ends = i + len - 1; // 结束位置
            for(int j = i; j < ends; j ++){ // 枚举每段划分可能
                dp[i][ends] = min(dp[i][ends], dp[i][j] + dp[j + 1][ends] + pre[ends] - pre[i - 1]);
            }
        }
    }
    cout << dp[1][n];
}
```

---

## 01背包

```c++
# include <bits/stdc++.h>
using namespace std;

vector<int> v; // 物品体积
vector<int> w; // 物品价值
void solve(void){
    int n, V; // 物品个数， 背包体积
    cin >> n >> V;
    // init
    v.resize(n + 1, 0); 
    w.resize(n + 1, 0);
    vector<vector<int> > dp(n + 1, vector<int>(V  + 1, 0));    
    for(int i = 1; i < i; i ++){
        cin >> v[i] >> w[i];
    }
    for(int i = 1; i <= n; i ++){ // 枚举物品
        for(int j = 1; j <= V; j ++){ // 枚举背包的体积
            dp[i][j] = dp[i - 1][j]; // 不取该物品
            if(j >= v[i]) dp[i][j] = max(dp[i][j], dp[i - 1][j - v[i]] + w[i]); // 如果剩下的体积足够，取最大值
        }
    }
    cout << dp[n][V] << endl;
}

signed main(void){
    ios::sync_with_stdio(0);
    int t = 1;
    //cin >> t;
    while(t --){
        solve();
    }
}
```

---

## 数位dp

```c++
int dp1[100][1000];
int dp2[100][1000];
int n1,n2,Max,Min;
string s1,s2;
int dfs1(int pos,int sum,int islimit,int leadzero){
    if(pos>=n1){
        return (!leadzero && sum>=Min && sum <=Max) ? 1 : 0;
        //当长度为最后一个的时候,前导不为0并且合法就+1；
    }
    //当前导不为0并且不受限时返回记忆化的值；
    if(!islimit && !leadzero && dp1[pos][sum]!=-1){
        return dp1[pos][sum];
    }
    int up = islimit ? s1[pos]-'0' : 9; //当数字受限时，上线为当前数位;
    int res=0;
    for(int i=0;i<=up;i++){
        //如果之前受限，现在到顶了就代表之后数位也将受限;
        //例如12345,执行11xxx的时候前面后面可以随便填充,但是12xxx的时候就要判断第三位受限了;
        //当前导为0并且当前数位为0,前导继续受限;
        res+= dfs1(pos+1,sum+i,islimit && i==up,leadzero && i==0);
        res%=mod;
    }
    //进行记忆化搜索；
    if(!islimit && !leadzero){
        dp1[pos][sum] = res;
    }
    return res;
}
```

## Kadane's算法_最大连续子序列和算法

> **$dp$**: 表示在当前位置结束的最大子数组和，初始值为数组的第一个元素。
> **$maxSum$**: 表示全局最大子数组和，初始值也为数组的第一个元素。

$$
dp\left[i\right] = 
\begin{cases}
a\left[i\right] &i = 1\\
\max \left(a\left[i\right], dp\left[i - 1\right] + a\left[i\right]\right) &1 < i \leq n
\end{cases}
$$
$$
maxSum =
\begin{cases}
a\left[i\right] &i = 1\\
\max \left(dp\left[i\right], maxSum\left[i - 1\right]\right) &1 < i \leq n
\end{cases}
$$

### 非限定区间

```c++
void solve(void){
    int n;
    cin >> n;
    vector<int> arr(n);
    vector<int> dp(n);
    int maxSum;
    for(int i = 0; i < n; i ++){
        cin >> arr[i];
    }
    maxSum = dp[0] = arr[0];
    for(int i = 1; i < n; i ++){
        dp[i] = max(arr[i], dp[i - 1] + arr[i]);
        maxSum = max(maxSum, dp[i]);
    }
    cout << maxSum << endl;
}
```

### 限定区间

```c++
void solve(void){
    int n, k;
    cin >> n >> k;
    vector<int> arr(n);
    vector<int> dp(n);
    int maxSum;
    for(int i = 0; i < n; i ++){
        cin >> arr[i];
    }
    maxSum = dp[0] = arr[0];
    int lf = 0, rt;
    for(rt = 1; rt < n; rt ++){
        dp[rt] = max(arr[rt], dp[rt - 1] + arr[rt]);
        if(rt - lf + 1 <= k){
            if(dp[rt] <= 0){
                maxSum = max(maxSum, dp[rt]);
                dp[rt] = 0;
                lf = rt + 1;
                continue;
            }
        }else{
            dp[rt] -= arr[lf];
            lf ++;
            while(arr[lf] < 0 && lf < rt){
                dp[rt] -= arr[lf];
                lf ++;
            }
        }
        maxSum = max(maxSum, dp[rt]);
    }
    cout << maxSum << endl;
}
```


---



---

# 区间查询

## 线段树

模板一

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define ll long long
#define LfTree(x) (x << 1)
#define RtTree(x) (x << 1 | 1)

struct node{
    ll val, lf, rt;
    node():val(-1), lf(-1), rt(-1){};
    node(ll _val, ll _lf, ll _rt):val(_val), lf(_lf), rt(_rt){};
};

ll N;
vector<node> tree; // 从1开始
vector<ll> tag; // 懒惰标记,lazy_tag
vector<ll> arr; // 存入数中的值

void build(ll inx, ll lf, ll rt){ // 建线段树
    if(lf == rt){
        tree[inx] = node(arr[lf], lf, rt); // 存储叶子节点
        return;
    }
    ll mid = lf + (rt - lf) / 2;
    build(LfTree(inx), lf, mid); // 左子树
    build(RtTree(inx), mid + 1, rt); // 右子树
    tree[inx] = node(tree[LfTree(inx)].val + tree[RtTree(inx)].val, lf, rt); // 根据左右子树确定该节点
}

void add_tag(ll inx, ll val){   // 添加懒惰标记
    tag[inx] += val;
    tree[inx].val += (tree[inx].rt - tree[inx].lf + 1) * val;
} 

void emplace_tag(ll inx){ // 传递懒惰标记
    if(!tag[inx]) return; // 没有懒惰标记
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    add_tag(LfTree(inx), tag[inx]); // 传递给左子树
    add_tag(RtTree(inx), tag[inx]); // 传递给右子树
    tag[inx] = 0; // 清除自己的标记
}

void update(ll inx, ll lf, ll rt, ll val){ // 修改数据
    if(lf <= tree[inx].lf && tree[inx].rt <= rt){ // 该节点被所求区间全覆盖，修改并添加懒惰标记
        add_tag(inx, val);
        return;
    }
    emplace_tag(inx); // 懒惰标记传递，确保左右子树为最新状态
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    if(lf <= mid){ // 更新左节点
        update(LfTree(inx), lf, rt, val);
    } 
    if(mid < rt){ // 更新右节点
        update(RtTree(inx), lf, rt, val);
    }
    tree[inx].val = tree[LfTree(inx)].val + tree[RtTree(inx)].val; //  更新自己
}

ll query(ll inx, ll lf, ll rt){ // 查询
    emplace_tag(inx); // 传递懒惰标记
    if(lf <= tree[inx].lf && tree[inx].rt <= rt){ // 该节点被所求区间全覆盖
        return tree[inx].val;
    }
    ll mid = tree[inx].lf + (tree[inx].rt - tree[inx].lf) / 2;
    ll ans = 0;
    if(lf <= mid){ // 左子树与所求区间有交集
        ans += query(LfTree(inx), lf, rt);
    }
    if(mid < rt){ // 右子树与所求区间有交集
        ans += query(RtTree(inx), lf, rt);
    }
    return ans;
}

void printTree(){ // 输出线段树每个节点的值
    cout << "修改后线段树每个节点的值：" << endl;
    for (ll i = 1; i < tree.size(); ++i) {
        if(tree[i].lf == -1) break;
        cout << "Node " << i << ": [" << tree[i].lf << ", " << tree[i].rt << "] = " << tree[i].val << endl;
    }
}

void init(){
    tree.resize(N * 4);
    arr.resize(N + 1);
    tag.resize(N * 4, 0);
    for(ll i = 1; i <= N; i ++){
        cin >> arr[i];
    }
    build(1, 1, N);
}
void solve(void){
    cin >> N;
    init();
}

signed main(void){
    ios::sync_with_stdio(false);
    solve();
    return 0;
}
```

模板二

```c++
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define pii pair<int, int>
#define get_lf(x) ((x) << 1)
#define get_rt(x) ((x) << 1 | 1)
const int N = (int)1e5 << 3;

int tree[N] = {0};  // 线段树
int lazy[N] = {0};  // 懒惰标记
int n; // 数据个数
vector<int> arr; // 基本数组

void add_tag(int inx, int lf, int rt, int val){ // 添加懒惰标记
    /*
        inx : 指向当前线段树索引位置。
        [lf, rt] : 线段树当前索引指代区间。
        val : 添加的懒惰标记值。
    */
    lazy[inx] += val;
    tree[inx] += (rt - lf + 1) * val;
}

void emplace_tag(int inx, int lf, int rt){  // 传递懒惰标记
    /*
        inx : 指向当前线段树索引位置。
        [lf, rt] : 线段树当前索引指代区间。
    */
    if(!lazy[inx]) return;
    int mid = lf + ((rt - lf) >> 1);
    add_tag(get_lf(inx), lf, mid, lazy[inx]);
    add_tag(get_rt(inx), mid + 1, rt, lazy[inx]);
    lazy[inx] = 0;
}

void build_tree(int inx, int lf, int rt){ // 建树
    /*
        inx : 指向当前线段树索引位置。
        [lf, rt] : 线段树当前索引指代区间。
    */
    if(lf == rt){
        tree[inx] = arr[lf];
        return;
    }
    int mid = lf + ((rt - lf) >> 1);
    build_tree(get_lf(inx), lf, mid);
    build_tree(get_rt(inx), mid + 1, rt);
    tree[inx] = tree[get_lf(inx)] + tree[get_rt(inx)];
}

int query_sum(int inx, int lf, int rt, int q_lf, int q_rt){ // 区间和
    /*
        inx : 指向当前线段树索引位置。
        [lf, rt] : 线段树当前索引指代区间。
        [q_lf, q_rt] : 需求和区间。
    */
    emplace_tag(inx, lf, rt);
    if(q_lf <= lf && rt <= q_rt){
        return tree[inx];
    }
    int sum = 0;
    int mid = lf + ((rt - lf) >> 1);
    if(mid >= q_lf) sum += query_sum(get_lf(inx), lf, mid, q_lf, q_rt);
    if(mid < q_rt) sum += query_sum(get_rt(inx), mid + 1, rt, q_lf, q_rt);
    return sum;
}

void update(int inx, int lf, int rt, int t_lf, int t_rt, int val){ // 区间更新数据（加）
    /*
        inx : 指向当前线段树索引位置。
        [lf, rt] : 线段树当前索引指代区间。
        [q_lf, q_rt] : 需更新区间。
        val : 更新值。
    */
    if(t_lf <= lf && rt <= t_rt){
        add_tag(inx, lf, rt, val);
        return; 
    }
    emplace_tag(inx, lf, rt);
    int mid = lf + ((rt - lf) >> 1);
    if(mid >= t_lf) update(get_lf(inx), lf, mid, t_lf, t_rt, val);
    if(mid < t_rt) update(get_rt(inx), mid + 1, rt, t_lf, t_rt, val);
    tree[inx] = tree[get_lf(inx)] + tree[get_rt(inx)];
}


void solve(){
    int m;
    cin >> n >> m;
    arr.resize(n + 1);
    for(int i = 1; i <= n; i ++){
        cin >> arr[i];
    }
    build_tree(1, 1, n);
    while(m --){
        int op, lf, rt, val;
        cin >> op >> lf >> rt;
        if(op == 1){
            cin >> val;
            update(1, 1, n, lf, rt, val);
        }else{
            cout << query_sum(1, 1, n, lf, rt) << endl;
        }
    }
}

signed main(void){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    int t = 1;
    // cin >> t;
    while(t --){
        solve();
    }
    return 0;
}
```

---

## ST算法

> ST算法通常用于求解 RMQ (Range-Minimum/Maximum Query, **区间最值问题**)  
> ST(Sparse-Table)算法：是一种用于解决 RMQ 问题的高效算法，它基于动态规划的思想，通过预处理得到一个二维数组，从而在O(1)的时间复杂度内回答  RMQ 查询。

### 模板(RMQ-max)

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
// #define int long long
#define ll long long
// 注：i^j 表示 i 的 j 次方

/*
注意数组长度，和是否需要开long long
*/

const int N = 1e5 + 10;
int n, q;   // 原数组长度，询问区间次数
int arr[N]; // 原数组
int dp[N][40]; // dp[i][j]对应从索引索引 i 开始，长度为 2^j 区间内的最值

// 初始化 st表
void st_init(){
    for(int i = 1; i <= n; i ++) dp[i][0] = arr[i];     // 初始化区间长度为 2^0 的值
    int p = (int)log2(n);           // 计算最长允许区间长度，避免 2^j 越界
    for(int k = 1; k <= p; k ++){   // 从低层向高层递推
        for(int s = 1; s + (1 << k) <= n + 1; s ++){    // 遍历区间初始位置，避免越界
            dp[s][k] = max(dp[s][k - 1], dp[s + (1 << (k - 1))][k - 1]); // 将低层两个区间最大值合并
        }
    }
}

// 获取区间的最大最小值
int st_query(int lf, int rt){
    int k = (int)log2(rt - lf + 1);                     // 计算最长允许区间长度
    return max(dp[lf][k], dp[rt - (1 << k) + 1][k]);    // 所求区间最大值
}

// 向原数组尾部添加新元素，并更新可能被影响的区间最值
void add_element(int value) {
    arr[++ n] = value;  // 将新数据添加到数组末尾
    // 更新 dp 数组
    dp[n][0] = value;   // 初始化新元素的区间长度为 2^0 的值
    int p = (int)log2(n);   // 计算当前允许的最大区间长度
    for (int k = 1; k <= p; k++) {
        int s = n - (1 << k) + 1;   // 计算新数据可能影响的起始位置
        if (s > 0) {
            dp[s][k] = max(dp[s][k - 1], dp[s + (1 << (k - 1))][k - 1]);
        }
    }
}

void solve(void){
    cin >> n >> q;
    st_init();
    while(q --){
        int lf, rt;
        cin >> lf >> rt;
        cout << st_query(lf, rt) << endl;
    }
}

signed main(void){
    ios::sync_with_stdio(false);
    // cin.tie(0);
    // cout.tie(0);
    int t = 1;
    // cin >> t;
    while(t --){
        solve();
    }

    return 0;
}
```

---

## 归并树

> 获取区间内大于或小于 x 的数量。

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define int long long
#define ll long long
#define LfTree(x) (x << 1)
#define RtTree(x) (x << 1 | 1)

struct node{
    vector<int> val;
    int lf, rt;
    node():lf(-1), rt(-1){};
    node(vector<int> _val, int _lf, int _rt):val(_val), lf(_lf), rt(_rt){};
};

int N;
vector<node> tree; // 从1开始
vector<int> arr; // 存入数中的值

void build(int inx, int lf, int rt){ // 建归并树
    if(lf == rt){
        tree[inx] = node({arr[lf]}, lf, rt); // 存储叶子节点
        return;
    }
    int mid = lf + (rt - lf) / 2;
    build(LfTree(inx), lf, mid); // 左子树
    build(RtTree(inx), mid + 1, rt); // 右子树
    // 将左右子树的有序数组拼接为新的有序数组
    merge(
        tree[LfTree(inx)].val.begin(), tree[LfTree(inx)].val.end(), 
        tree[RtTree(inx)].val.begin(), tree[RtTree(inx)].val.end(),
        back_inserter(tree[inx])
    );
}

int query_less(int inx, int lf, int rt, int x){ // 获取区间[lf,rt]中小于 x 的元素数量
    if(tree[inx].lf > rt || tree[inx].rt < lf) return 0;
    if(tree[inx].lf >= lf && tree[inx].rt <= rt){ // 该节点完全被所求区间包含
        auto it = lower_bound(tree[inx].val.begin(), tree[inx].val.end(), x);
        return it - tree[inx].val.begin();  // 返回该节点所表示区间中满足条件的元素数量
    }
    int mid = tree[inx].lf + ((tree[inx].rt - tree[inx].lf) >> 1);
    return query_less(LfTree(inx), lf, rt, x) + query_less(RtTree(inx), lf, rt, x); // 将左右子树的结果相加
}

int query_greater(int inx, int lf, int rt, int x){  // 获取区间[lf,rt]中大于 x 的元素数量
    if(tree[inx].lf > rt || tree[inx].rt < lf) return 0;
    if(tree[inx].lf >= lf && tree[inx].rt <= rt){   // 该节点完全被所求区间包含
        auto it = upper_bound(tree[inx].val.begin(), tree[inx].val.end(), x);
        return tree[inx].val.end() - it;    // 返回该节点所表示区间中满足条件的元素数量
    }
    int mid = tree[inx].lf + ((tree[inx].rt - tree[inx].lf) >> 1);
    return query_greater(LfTree(inx), lf, rt, x) + query_greater(RtTree(inx), lf, rt, x); // 将左右子树的结果相加
}

void init(){
    tree.reserve(N * 4);
    arr.resize(N + 1);
    for(int i = 1; i <= N; i ++){
        cin >> arr[i];
    }
    build(1, 1, N);
}
void solve(void){
    cin >> N;
    init();
}

signed main(void){
    ios::sync_with_stdio(false);
    solve();
    return 0;
}
```

---

## 树状数组

> **求区间和。** 可以在 $O(nlogn)$ 的时间构建树状数组， $O(logn)$ 的时间更新指定节点数据和查询区间和。

```c++
const int N = 1e5 + 10;
int lowbit(int x){return x & -x;} // 获取二进制最后一个1后的数值，例如 110 -> 10 也就是6 -> 2
/*
注意！！！
tree需要从索引 1 开始，否则 inx += lowbit(inx); inx 会进入死循环
*/
int tree[N];
void add(int inx, int val){ // 更新元素a[x] += d;
    while(inx <= N){
        tree[inx] += val;
        inx += lowbit(inx);
    }
}
int get_pre(int x){ // 返回前缀和
    int ans = 0;
    while(x > 0){
        ans += tree[x];
        x -= lowbit(x);
    }
    return ans;
}
int sum(int lf, int rt){ // 获取[lf, rt]的范围和
    return get_pre(rt) - get_pre(lf - 1);
}
```

---

# 二分

## 基础模板

**模板一：**

```c++
int binarySearch(vector<int>& nums, int target){
    // 返回所求元素是否在 nums 中出现，返回目标索引，否则返回 -1
    if(nums.size() == 0){
        return -1;
    }
    int lf = 0, rt = nums.size() - 1;
    while(lf <= rt){
        // Prevent (lf + rt) overflow
        int mid = lf + (rt - lf) / 2; // or mid = lf + ((rt - lf) >> 1);
        if(nums[mid] == target){ // 找到目标
            return mid; // 返回目标的索引
        }else if(nums[mid] < target){
            lf = mid + 1; 
        }else{
            rt = mid - 1; 
        }
    }
    // End Condition: lf > rt
    return -1;
}
```

**模板二：**

```c++
int binarySearch(vector<int>& nums, int target){
    if(nums.size() == 0){
        return -1;
    }
    int lf = 0, rt = nums.size();
    while(lf < rt){
        // Prevent (lf + rt) overflow
        int mid = lf + (rt - lf) / 2; // or mid = lf + ((rt - lf) >> 1);
        if(nums[mid] == target){
            return mid; 
        }else if(nums[mid] < target){
            lf = mid + 1; 
        }else{ 
            rt = mid;
        }
    }
    // Post-processing:
    // End Condition: lf == rt
    if(lf != nums.size() && nums[lf] == target) return lf;
    return -1;
}
```

**模板三：**

```c++
int binarySearch(vector<int>& nums, int target){
    if (nums.size() == 0){
        return -1;
    }
    int lf = 0, rt = nums.size() - 1;
    while (lf + 1 < rt){
        // Prevent (lf + rt) overflow
        int mid = lf + (rt - lf) / 2; // or mid = lf + ((rt - lf) >> 1);
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            lf = mid;
        } else {
            rt = mid;
        }
    }
    // Post-processing:
    // End Condition: lf + 1 == rt
    if(nums[lf] == target) return lf;
    if(nums[rt] == target) return rt;
    return -1;
}
```

---

## 实数二分

> [一元三次方程求解](https://www.lanqiao.cn/problems/764/learning/?page=1&first_category_id=1&problem_id=764)

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

double a, b, c, d; // 多项式各系数
double handle(double x){ // 计算所像是所得值
    return a * pow(x, 3) + b * pow(x, 2) + c * x + d;
}

void solve(void){
    cin >> a >> b >> c >> d;
    for(int i = -100; i < 100; i ++){ // 题目给出的答案范围
        double x1 = i, x2 = i + 1;  // 假设连个解
        double y1 = handle(x1), y2 = handle(x2); // x1,x2带入得到的y值
        if(y1 == 0){ // x1为所求解之一
            cout << fixed << setprecision(2) << x1 << " ";
            continue;
        }
        if(y1 * y2 >= 0) continue;
        for(int j = 0; j < 100; j ++){ // 二分答案，不断逼近正确答案
            double mid = (x1 + x2) / 2.0;
            if(handle(mid) * handle(x2) <= 0){
                x1 = mid;
            }else{
                x2 = mid;
            }
        }
        cout << fixed << setprecision(2) << x2 << " ";
    }
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

---

# 三分法




---

# 单调栈

```c++
vector<int> pre_g(n); // 该元素 arr[i] 左侧最接近且严格大于自己的索引位置
stack<int> st; // 单调栈
for(int i = 0; i < n; i ++){
    while(!st.empty() && arr[st.top()] <= arr[i]){  // 非空且栈顶小于等于自身
        st.pop();
    }
    pre_g[i] = st.empty() ? -1 : st.top();
    st.emplace(i);
}
```

# 单调双端队列

```c++
vector<int> window_max(n);
deque<int> deq;
for(int i = 0; i < n; i ++){
    while(!deq.empty() && arr[deq.back()] <= arr[i]){
        deq.pop_back();
    }
    while(!deq.empty() && deq.front() <= i - k){
        deq.pop_front();
    }
    deq.emplace(i);
    window_max[i] = arr[deq.front()];
}
```

---

# 排序

## 归并排序

> **时间复杂度：$O(n \; logn)$**

```c++
const int N = 1e5 + 10;

int arr[N]; // 数据数组
int help[N];    // 辅助排序数组

void merge(int lf, int mid, int rt){
    int i = lf;
    int a = lf, b = mid + 1;
    while(a <= mid && b <= rt){
        help[i ++] = arr[a] <= arr[b] ? arr[a ++]: arr[b ++];
    }
    while(a <= mid){
        help[i ++] = arr[a ++];
    }
    while(b <= rt){
        help[i ++] = arr[b ++];
    }
    for(i = lf; i <= rt; i ++){
        arr[i] = help[i];
    }
}

void merge_sort(int lf, int rt){
    if(lf == rt) return;
    int mid = lf + ((rt - lf) >> 1);
    merge_sort(lf, mid);
    merge_sort(mid + 1, rt);
    merge(lf, mid, rt);
}
```

### 归并分治

1. 考虑一个问题在大范围上的答案，是否等于，左部分的答案 + 右部分的答案 + 跨越左右产生的答案。
2. 计算"跨域左右产生的答案"时，如果加上左、右各自有序这个设定，会不会获得计算的便利性。
3. 如果以上两点都成立，那么该问题很可能被归并分治解决。



---

# 随记

## 全排列

### 普通全排列

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4
};

void dfs(int s, int t){ // 从第s个数开始到第k个数结束的全排列
    if(s == t){
        for(int i = 0; i <= t; i ++) cout << target[i] << " ";  // 输出一个排列
        cout << endl;
        return;
    }
    for(int i = s; i <= t; i ++){
        swap(target[s], target[i]); // 第1个数和后面的数交换
        dfs(s + 1, t);
        swap(target[s], target[i]); // 回溯
    }
}
/*
1 2 3
1 3 2 
2 1 3 
2 3 1 
3 2 1 
3 1 2
*/

void solve(void){
    int n = target.size();
    dfs(0, n - 1);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

### 从小到大输出排列

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4, 5
};
vector<bool> vis(20, false);    //第i个数是否被用过
vector<int> res(20, 0);    // 生成的一个全排列

void dfs(int s, int t){ // 从第s个数开始到第k个数结束的全排列
    if(s == t){
        for(int i = 0; i < t; i ++) cout << res[i] << " ";  // 输出一个排列
        cout << endl;
        return;
    }
    for(int i = 0; i < t; i ++){
        if(!vis[i]){
            vis[i] = true;
            res[s] = target[i];
            dfs(s + 1, t);
            vis[i] = false;
        }
    }
}

void solve(void){
    int n = target.size();
    dfs(0, n);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}          
```

---

## 组合

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'

vector<int> target = {
    1, 2, 3, 4, 5
};
vector<bool> vis(20, false);    //第i个数是否被用过

void dfs(int k, int n){ // dfs到k个数
    if(k == n){
        for(int i = 0; i < n; i ++){
            if(vis[i]) cout << target[i] << '-';
        }
        cout << endl;
        return;
    }
    vis[k] = false; // 不选第k个数
    dfs(k + 1, n);
    vis[k] = true;  // 选这个数
    dfs(k + 1, n);
}

void solve(void){
    int n = target.size();
    dfs(0, n);
}

signed main(void){
    ios::sync_with_stdio(0);
    //cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

---

## 高精度加法

```c++
string add(string a, string b){
    string s = "";
    int op = 0;
    for(int i = a.size() - 1, j = b.size() - 1; i >= 0 || j >= 0 || op > 0; i --, j --){
        if(i >= 0) op += a[i] - '0';
        if(j >= 0) op += b[j] - '0';
        s += to_string(op % 10);
        op /= 10;
    }
    reverse(s.begin(), s.end());
    return s;
}
```

---

## 阶乘

```c++
const int N = 1e4;
vector<int> arr;

void func(int n){
    arr.resize(N, 0); 
    arr[0] = 1;
    for(int i = 1; i <= n; i ++){
        int op = 0;
        for(int j = 0; j < N; j ++){
            arr[j] = arr[j] * i + op;
            op = arr[j] / 10;
            arr[j] = arr[j] % 10;
        }
    }
    // show;
    int last;
    for(int i = N - 1; i >= 0; i --){
        if(arr[i] != 0){
            last = i;
            break;
        }
    }
    for(int i = last; i >= 0; i --) cout << arr[i];
}
```

---

## 除法模拟

>**题目**：
这里所谓的“光棍”，并不是指单身汪啦~ 说的是全部由1组成的数字，比如1、11、111、1111等。传说任何一个光棍都能被一个不以5结尾的奇数整除。比如，111111就可以被13整除。 现在，你的程序要读入一个整数x，这个整数一定是奇数并且不以5结尾。然后，经过计算，输出两个数字：第一个数字s，表示x乘以s是一个光棍，第二个数字n是这个光棍的位数。这样的解当然不是唯一的,题目要求你输出最小的解。
**提示**：一个显然的办法是逐渐增加光棍的位数，直到可以整除x为止。但难点在于，s可能是个非常大的数 —— 比如，程序输入31，那么就输出3584229390681和15，因为31乘以3584229390681的结果是111111111111111，一共15个1。
**输入格式：**
输入在一行中给出一个不以5结尾的正奇数x（<1000）。
**输出格式：**
在一行中输出相应的最小的s和n，其间以1个空格分隔。
**输入样例：**
31
**输出样例：**
3584229390681 15

```c++
#include<stdio.h>
int main(){
    int n, r = 1, w = 1;//r表示1,11,111类型的数据，w记录位数
    scanf("%d", &n);
    while(r < n){
        r *= 10;
        r++;
        w++;
    }
    while(1){
        printf("%d", r/n);//输出商
        r %= n;//取余
        if(r == 0)//取余后，若等于0，则证明能被整除，break掉
            break;
        r = r * 10 + 1;//不等于0则在余数后一位加上1
        w++;
    }
    printf(" %d",w);
    return 0;
} 
```

other  

```c++
int a, b, e, len = 0; cin >> a >> b >> e;
string res = "";
res += to_string(a/b) + ".";
a = a % b;
a *= 10;
while(a < b){
    len ++;
    res += "0";
    a *= 10;
    if(len > e){
        res.pop_back();
        cout << res;
        return;
    }
}
while(true){
    res += to_string(a / b);
    a %= b;
    len ++;
    if(len > e){
        if(res.back() >= '5'){
            int op = 1, inx = res.size() - 2;
            res.pop_back();
            int flag;
            while(op == 1){
                if(res[inx] == '.') inx --;
                flag = res[inx] - '0' + op;
                op = flag / 10;
                res[inx] = '0' + flag % 10;
                inx --;
                if(op == 1 && inx == -1){
                    res = "1" + res;
                    break;
                }
            }
        }else{
            res.pop_back();
        }
        cout << res << endl;
        return;
    }
    a *= 10;
}
```

---

## 后缀表达式（逆波兰式）

**定义：** 指不包含括号，运算符放在两个运算对象的后面，所有的运算按运算符出现的顺序，严格从左向右进行（不考虑运算符的优先级规则）。  

**计算：** 从左往右扫描表达式，遇到数字时，将数字压入栈中，遇到运算符时，弹出栈顶的两个数，用运算符对他们做相应的计算，并将结果入栈。  

**例：** 2 3 + 4 * 5 -  ==> 15  

**注：** 由于后缀表达式忽略了括号，所以在转化为中缀表达式后主意括号的影响。

```c++
# include <bits/stdc++.h>
using namespace std;
#define endl '\n'

// 优先级， 越高优先级越高
map<char, int> priority = {
    {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'(', 0}, {')', 0}
};

// 比较运算符的优先级
bool judge(char a, char b){ 
    // a <= b;
    return priority[a] <= priority[b];
}

// 将中序表达式转换为后序表达式
string getPostfix(string infix){
    // 操作数栈和运算符栈
    stack<string> operands;
    stack<char> operators;
    string num = "";
    for(auto ch : infix){
        // 获取数字
        if(isdigit(ch)){ 
            num.push_back(ch);
            continue;
        }
        // 将数字压入操作数栈
        if(!num.empty()){ 
            operands.emplace(num);
            num.clear();
        }
        // 左括号压入运算符栈中
        if(ch == '('){ 
            operators.emplace(ch);
            continue;
        }
        // 右括号
        if(ch == ')'){
            // 弹出运算符栈顶元素，直至遇到左括号
            while(!operators.empty() && operators.top() != '('){
                operands.emplace(string(1, operators.top()));
                operators.pop();
            }
            // 弹出剩余的左括号
            if(!operators.empty()) operators.pop();
            continue;
        }
        if(operators.empty() || operators.top() == '(' || !judge(ch, operators.top())){
            /*
            如果非括号，当运算符栈为空，或者运算符栈顶为左括号，
            或者比运算符栈顶的优先级高，将当前运算符压入运算符栈
            */
            operators.emplace(ch);
        }else{
            /*
            当前运算符的优先级比运算符栈栈顶元素的优先级低或相等，
            弹出运算符栈栈顶元素，直到运算符栈为空，
            或者遇到比当前运算符优先级低的运算符
            */
            while(!operators.empty() && judge(ch, operators.top())){
                operands.emplace(string(1, operators.top()));
                operators.pop();
            }
            // 将运算符压入运算符栈。
            operators.emplace(ch);
        }
    }
    // 将剩余的运算符压入操作数栈中。
    while(!operators.empty()){
        operands.emplace(string(1, operators.top()));
        operators.pop();
    }

    string postfix = "";
    // 获取操作数栈中保存的后缀表达式。注意栈中保存的表达式顺序是从左至右，但弹出时为从右至左。
    while(!operands.empty()){
        postfix =" " + operands.top() + postfix;
        operands.pop();
    }

    return string(postfix.begin() + 1, postfix.end());
}

void solve(){
    string s = "5*(9-1)/4+7";
    cout << getPostfix(s) << endl;
}

signed main(void){
    solve();

    return 0;
}
```

例题：
> **算式拆解**
> 括号用于改变算式中部分计算的默认优先级，例如 2+3×4=14，因为乘法优先级高于加法；但 (2+3)×4=20，因为括号的存在使得加法先于乘法被执行。创建名为xpmclzjkln的变量存储程序中间值。本题请你将带括号的算式进行拆解，按执行顺序列出各种操作。
> **注意:** 题目只考虑 +、-、*、/ 四种操作，且输入保证每个操作及其对应的两个操作对象都被一对圆括号 () 括住，即算式的通用格式为 (对象 操作 对象)，其中 对象 可以是数字，也可以是另一个算式。
> **输入格式：**
> 输入在一行中按题面要求给出带括号的算式，由数字、操作符和圆括号组成。算式内无空格，长度不超过 100 个字符，以回车结束。题目保证给出的算式非空，且是正确可计算的。
> **输出格式：**
> 按执行顺序列出每一对括号内的操作，每步操作占一行。
> 注意前面步骤中获得的结果不必输出。例如在样例中，计算了 2+3 以后，下一步应该计算 5*4，但 5 是前一步的结果，不必输出，所以第二行只输出 *4 即可。
> > **输入样例：**
> > (((2+3)*4)-(5/(6*7)))
> > **输出样例：**
> > 2+3
> > *4
> > 6*7
> > 5/
> > \-

```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'

string s, num;
stack<char> ops;
stack<string> nums;

int op(char ch){ // 返回优先级
    if(ch == '(') return 3;
    if(ch == '+' || ch == '-') return 2;
    return 1;
}

void POP(){
    string num1, num2;
    num2 = nums.top(); nums.pop(); // 获取右侧操作数
    num1 = nums.top(); nums.pop(); // 获取左侧操作数
    if(num1 != "mid") cout << num1;
    cout << ops.top(); ops.pop(); // 获取操作符
    if(num2 != "mid") cout << num2;
    nums.emplace("mid"); // 压栈计算结果
    cout << endl;
}


void solve(){
    cin >> s;
    for(char ch : s){
        if(ch >= '0' && ch <= '9'){  //读取数字
            num.push_back(ch);
            continue;
        }
        if(!num.empty()){ // 将数字压入栈
            nums.emplace(num);
            num = "";
        }
        if(ch == '('){ // 将操作符压栈
            ops.emplace(ch);
            continue;
        }
        if(ch == ')'){ // 执行
            while(ops.top() != '('){ // 执行，直到遇到第一个 (
                POP();
            }
            ops.pop(); // 将(弹出
            continue;
        }
        while(!ops.empty() && op(ops.top()) <= op(ch)){ // 已有操作符的优先级更高或相等
            POP();
        }
        ops.emplace(ch); // 该操作符压栈
    }
    while(!ops.empty()){ // 执行完剩下的操作
        POP();
    }
    return;
}

signed main(void){
    ios::sync_with_stdio(false);
    int t = 1;
// cin >>t;
    while(t --){
        solve();
    }

    return 0;
}

```

---

## 快速幂

```c++
int fastPow(int a, int n){ // a^n
    int ans = 1;
    while(n){
        if(n & 1) ans *= a;
        a *= a;
        n >>= 1;
    }
    return ans;
}

int modPow(int a, int n, int mod){
    int ans = 0;
    while(n){
        if(n & 1) ans = ans * a % mod;
        a = a * a % mod;
        n >>= 1;
    }
    return ans;
}
```

或是

```c++
int fastPow(int a, int n){
    if(n == 0) return 1;
    int flag = fastPow(int a * a, n >> 1);
    if(n & 1) flag *= a;
    return flag;
}

int modPow(int a, int n, int mod){
    if(n == 0) return 1;
    int flag = modPow(int a * a % mod, n >> 1, mod);
    if(n & 1) flag = flag * a % mod;
    return flag;
}
```

---


## 整除分块

> 对于给定的正整数 n ,  n = k * i + r (0 <= r < i)。当 i 在一定范围内变化时，$\lfloor\frac{n}{i}\rfloor$(向下取整)会有很多重复情况。例如：$n = 10$ 时，$\lfloor\frac{10}{1}\rfloor = 10$，$\lfloor\frac{10}{2}\rfloor = 5$，$\lfloor\frac{10}{3}\rfloor=\lfloor\frac{10}{4}\rfloor = 2$，$\lfloor\frac{10}{5}\rfloor = 2$，$\lfloor\frac{10}{6}\rfloor=\lfloor\frac{10}{7}\rfloor=\lfloor\frac{10}{8}\rfloor=\lfloor\frac{10}{9}\rfloor=\lfloor\frac{10}{10}\rfloor = 1$。可以发现，$\lfloor\frac{n}{i}\rfloor$ 的值会呈现出块状分布的特点，相同值的 $i$ 会形成一个块。

### 算法实现

- 假设要计算$\sum_{i = 1}^{n}\lfloor\frac{n}{i}\rfloor$，可以通过整除分块来优化计算。
- 对于每个块，设块的左端点为 $l$，右端点为 $r$。当 $i = l$ 时，$\lfloor\frac{n}{l}\rfloor$ 的值确定，而该块的右端点 $r$ 可以通过 $r=\lfloor\frac{n}{\lfloor\frac{n}{l}\rfloor}\rfloor$ 计算得出。这样就可以在 $O(\sqrt{n})$ 的时间复杂度内计算出上述求和式子的值。

#### 公式

$r=\lfloor\frac{n}{\lfloor\frac{n}{l}\rfloor}\rfloor$

### 代码示例

```cpp
#include <iostream>
#include <cmath>

using namespace std;

int main() {
    int n;
    cin >> n;
    int ans = 0;
    for (int l = 1, r; l <= n; l = r + 1) {
        // 计算当前块的右端点
        r = n / (n / l);
        // 累加当前块的值
        ans += (r - l + 1) * (n / l);
    }
    cout << ans << endl;
    return 0;
}
```

---

## 常用集合算法

### 交集 set_intersection

set_intersection 函数用于求两个**有序序列**的交集，并将交集结果存储到一个**新的容器**中。  
返回一个迭代器，指向交集结果在新容器中的**最后一个元素的下一个位置**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_intersection( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first
                );

// first1 和 last1：表示第一个有序序列的起始和结束迭代器。
// first2 和 last2：表示第二个有序序列的起始和结束迭代器。
// d_first：表示存储交集结果的目标容器的起始迭代器。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    // 输入的容器必须是有序的。
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间, 否则会报错
    result.resize(min(vec1.size(), vec2.size()));

    // 计算交集
    auto it = set_intersection(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    // 调整结果容器的大小以匹配实际的交集元素数量
    result.resize(it - result.begin());

    cout << "交集: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}    
```

### 并集 set_union

set_union 函数用于求两个**有序序列**的并集，并将并集结果**存储到一个新的容器中**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_union( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first 
                );
// 参数含义与 set_intersection 类似。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间
    result.resize(vec1.size() + vec2.size());

    auto it = set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    result.resize(it - result.begin());

    cout << "并集: ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}

```

### 差集 set_difference

set_difference 函数用于求两个**有序序列**的差集（即第一个序列中存在而第二个序列中不存在的元素），并将差集结果**存储到一个新的容器中**。

```c++
template< class InputIt1, class InputIt2, class OutputIt >
OutputIt set_difference( 
                InputIterator1 first1, 
                InputIterator1 last1,
                InputIterator2 first2, 
                InputIterator2 last2,
                OutputIterator d_first 
                );
// 参数含义与前面两个函数类似。
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> vec1 = {1, 3, 5, 7, 9};
    vector<int> vec2 = {2, 3, 4, 5, 6};
    vector<int> result;

    // 为结果容器预留足够的空间
    result.resize(vec1.size() + vec2.size());

    auto it = set_difference(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), result.begin());
    result.resize(it - result.begin());

    cout << "差集 (vec1 - vec2): ";
    for (int num : result) {
        cout << num << " ";
    }
    cout << std::endl;

    return 0;
}

```
