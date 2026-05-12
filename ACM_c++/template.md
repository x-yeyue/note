# 数论

## 定理

### 算数基本定理(唯一分解定理)

> **算数基本定理**($fundamental ~ theorem ~ of ~ arithmetic$, 又称**唯一分解定理**，$unique ~ factorization ~ theorem$) —— 任意**合数** $c$ 可以表示成有限个素数 $p_{i}$ 的乘积，而且在不考虑素数乘积顺序的情况下其表示方式唯一。
> $$c = p_{1}^{e_{1}} p_{2}^{e_{2}} \cdots p_{n}^{e_{n}}$$

**引理：** 对任意大于 $1$ 的正整数 $n$ ，$n$ 至多有一个大于 $\sqrt{n}$ 的质数。假设该质数存在，它的唯一分解式子中的质数一定为 $1$ 。

### 费马小定理

> **费马小定理**($Fermat's ~ little ~ theorem$)指出：给定任意**素数** $p$，对于**整数** $a$,如果 $1 \lt a \lt p$，有以下结论：
> $$a^{p-1} \equiv 1 \pmod{p}$$

### 裴蜀定理

> 如果 $a$ 和 $b$ 是不全为 $0$ 的整数，则有整数 $x$、$y$，使得 $ax + by = \gcd\left(a, b\right)$。

**推论**：

1. 如果 $a$ 和 $b$ 是不全为 $0$ 的整数，且 $a$ 和 $b$ 互质，当且仅当存在整数 $x$、$y$，使得 $ax+by=1$ 。
    - 若 $ax+by=1$，则 $a$ 和 $b$ 互质。
2. 如果 $a$ 和 $b$ 是不全为 $0$ 的整数，并且 $ax + by = c$ 有整数解解，那么 $c$ 一定是 $\gcd\left(a,b\right)$ 的整数倍。
    - 若 $c$ 不是 $\gcd\left(a,b\right)$ 的整数倍，那么 $ax + by = c$ 就没有整数解。
3. $a$ 和 $b$ 两项的裴蜀定理，可以推广到多项的情况。
    - 例如: $ax + by + cz + dp=\gcd\left(a,b,c,d\right) = s$

**注**：如果 $ax + by = c$一旦有解，就意味着一定有无穷多组 $\left(x, y\right)$，都可以使得式子成立。

## 整除性

> 如果 $a$ 和 $b$ 为整数且 $a \neq 0$，$a$ 整除($divides$) $b$ 是指存在整数 $c$ 使得 $b = ac$，如果 $a$ 整除 $b$，称 $a$ 是 $b$ 的一个因子，且称 $b$ 是 $a$ 的倍数，将其记为 $a \mid b$，如果 $a$ 不能整除 $b$，则将其记为 $a \nmid b$。

##  线性筛

```c++
bool judge[maxn];
int prime[maxn], cnt = 0;
int getPrimes(int n){
    for(int i = 2; i <= n; i ++){
        if(!judge[i]){
            prime[cnt ++] = i;
        }
        for(int j = 0; prime[j] * i <= n; j ++){
            judge[prime[j] * i] = true;
            if(i % prime[j] == 0){
                break;
            }
        }
    }
    return cnt;
}
```

## gcd 与 lcm

### 最大公约数(gcd)

> - **定义:** 对于不全为 0 的整数 $a$, $b$, 它们的最大公约数是 *最大的正整数* *d*, 使得 $d \mid a$ 且 $d \mid b$, 记作 $\gcd(a, b)$。
> - **约定:** $\gcd(a, 0) = |a|$ 

#### 欧几里得算法（辗转相除法）

$$
\gcd(a, b) = 
\begin{cases}
    a, & b = 0 \\ 
    \gcd \left(b, a \bmod b\right), & b \neq 0
\end{cases}
$$


```c++
int gcd(int a, int b){
    if(a < b) swap(a, b);
    return b ? gcd(b, a % b) : a;
}
```

也可以使用头文件 `<algorithm>` 中的内齿求最大公因数的函数`__gcd(a, b)`。

### 最小公倍数(lcm)

> **定义:** 对于正整数 $s, b$， 它们的最小公倍数是 *最小的正整数* $m$，使得 $a \mid m$ 且 $b \mid m$，记作 $lcm(a, b)$。

### 常用结论

1. 对于正整数 $a, b$​，有:
    $$
    \gcd(a, b) \times lcm(a, b) = a \times b
    $$

2. 根据 [唯一分解定理](#算数基本定理(唯一分解定理)) 得到:
    $$
    a = \prod p_{i}^{e_{i}}, ~ ~ b = \prod p_{i}^{f_{i}}
    $$
    则: 

    - $gcd(a, b) = \prod p_{i}^{\min(e_{i}, f_{i})}$
        取每个质因子指数的*最小值*。
    - $lcm(a, b) = \prod p_{i}^{\max(e_{i}, f_{i})}$
        区域每个质因子指数的*最大值*。

    **推论:** 

    - $gcd(a, b) ~|~ a ~,~ gcd(a, b) ~|~ b$
    - $a ~|~ lcm(a, b) ~,~ b ~|~ lcm(a, b)$
    - $a \times b = gcd(a, b) \times lcm(a, b)$ 就是 $min + max = e_{i} + f_{i}$ 的体现。

3. $gcd(a, b) = gcd(a, b - a)$
    可以推广到 $gcd(a_{1}, a_{2}, \dots, a_{n}) = gcd(a_1, a_{2} - a_{1}, \dots, a_{n} - a_{1})$
    **应用:**
    $$
    \begin{split}
    gcd(a_{1} + x, a_{2} + x, \dots, a_{n} + x) &= gcd(a_{1} + x, (a_{2} + x) - (a_{1} + x), \dots, (a_{n} + x) - (a_{n - 1} + x))\\
    &= gcd(a_{1} + x, a_{2} - a_{1}, \dots, a_{n} - a_{n - 1})
    \end{split}
    $$

### 互质

> 如果正数 $a$ 和 $b$ 的最大公因数为 $1$，则称 $a$ 和 $b$ **互素($relative ~ prime$，或称互质)**。

- 显然两个素数的最大公约数为 $1$，有时两个非质数的最大公约数也可能为 $1$，如 $4$ 和 $9$。
- 大于 $1$ 的两个质数总是互质的。

**互质的性质**
设 $a \lt b$，则 $ka\left(1 \leq k \leq b\right)$ 除以 $b$ 的余数会取遍 $0 \sim b-1$ 且不会发生重复。
例如： $5$ 和 $7$ 互为质数，则 $5k(1 \leq k \leq 7)$ 除以 $7$ 的余数依次为 $5$、$3$、$1$、$6$、$4$、$2$、$0$，取遍了 $0 \sim 6$ 的余数值。

## 模算术

$a \bmod b = c$ 即 $a = b \cdot k + c,(k \in \mathbb{Z})$

特别的，如果数 $a$ 和数 $b$ 关于 $m$ 的模相等，记作 $a \equiv b \pmod{m}$ 

### 模运算规则

**加法规则** : $(x + y) \bmod n = ((x \bmod n) + (y \bmod n)) \bmod n$  

**减法规则** : $(x - y) \bmod n = ((x \bmod n) - (y \bmod n)) \bmod n$  

**乘法规则** : $xy \bmod n = (x \bmod n)(y \bmod n) \bmod n$

**乘方规则** : $x ^ y \bmod n = (x \bmod n)^y \bmod n$

#### 结论

1. 判断一个数是否能被 $3$ 整除，只需要验证该整数各位数相加之和能否被 $3$整除即可。

根据模运算规则，有同余式 $10 \equiv 1 \pmod{3}$成立，因此有$10^k \equiv 1 \pmod{3}$成立，则有：

$$
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{3}
\end{split}
$$

2. 同样的，检验一个整数能否被 $9$ 整除，只需要检验该整数各位数相加之和能否被 $9$ 整除即可。

由于 $10 \equiv 1 \pmod{9}$ 成立，因此有 $10^k \equiv 1 \pmod{9}$ 成立,则有

$$
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2} a_{1} a_{0})_{10}&= a_{k}10^{k} + a_{k-1}k^{k-1} + \cdots + a_{1}10 + a_{0} \\
    &\equiv a_{k} + a_{k-1} + \cdots + a_{1} + a_{0} \pmod{9}
\end{split}
$$

3. 类似的因为 $10 \equiv -1 \pmod{11}$，有


$$
\begin{split}
    (a_{k}a_{k-1} \cdots a_{2}a_{1}a_{0})_{10} &= a_{k}10^{k}+a_{k-1}10^{k-1}+ \cdots + a_{1}10+a_{0} \\
    &\equiv a_{k}\left(-1\right)^{k} + a_{k-1}\left(-1\right)^{k-1}+ \cdots +a_{2} - a_{1} + a_{0} \pmod{11}
\end{split}
$$

这表明$(a_{k}a_{k-1} \cdots a_{2}a_{1}a_{0})_{10}$ 能被 $11$ 整除的充要条件是对 $n$ 的各位数字交替相加减，所得到的整数 $a_{0}-a_{1}+a_{2}-\cdots+\left(-1\right)^{k}a^{k}$能被 $11$ 整除。

4. 对第 $3$ 条进行推广可以得到 $k \equiv -1 \pmod{k + 1}$ 。

### 模的逆元

> **定义：**
> 若存在正整数 $a$、$b$、$m$，满足 $a \times b \equiv 1 \pmod{m}$，则称 $b$ 为 $a$ 在模 $m$ 下的逆元，一般记作 $a^{-1} \equiv b \pmod{m}$。
>
> **性质：**
>
> - **逆元是"相互"的:** 如果 $b$ 是 $a$ 的逆元，那么 $a$ 也是 $b$ 的逆元，即 $b^{-1} \equiv a \pmod{m}$
> - **把除法变乘法:** 若 $a \times b \equiv 1 \pmod{m}$，即 $b$ 为 $a$ 在模 $m$ 下的逆元，则 $x \div a \pmod{m}$ 等价于 $x \times b \pmod{m}$。

#### 费马小定理法

根据费马小定理，对于任意素数 $p$，如果整数 $a$ 满足 $1 \lt a \lt p$，有
$$a^{-1} \equiv 1 \pmod{p}$$

根据上述结论，有 $a^{p-1} \equiv a \times a^{p-2} \equiv 1 \pmod{p}$，则 $a$ 的逆元为 $a^{p-2}$。
**注：$p$需是素数。**

#### 扩展欧几里得法

给定模数 $p$，求 $a$ 模 $p$ 的逆元相当于求解 $ax \equiv 1\pmod{p}$，该线性同余方程可转为求解不定方程 $ax + ny = 1$，可利用扩展欧几里得算法求解，并将求解出的 $x_0$ 调整到区间$\left[0, n - 1\right]$ 即为所求逆元。
若 $\gcd\left(a, n\right) \neq 1$，则 $a$ 模 $n$ 的逆元不存在。

```c++
void extgcd(int a, int b, int &x, int &y){
    if(b == 0){
        x = 1, y = 0;
        return;
    }
    extgcd(b, a % b, x, y);
    int t = x - a / b * y;
    x = y, y = t;
}

int get_inv(int a, int p){
    // ax = 1 (mod p)
    if(gcd(a, p) != 1){
        return -1; // 无逆元
    }
    int x, y;
    extgcd(a, p, x, y);
    if(x < 0){
        x = (x % p + p) % p;
    }
    return x % p;
}
```

## 组合数学

### 排列与组合

> **排列**
> $$P\left(n, k\right) = \frac{n!}{\left(n - k\right)!}, 0 \le n, 0 \le k \le n$$
>
> 特殊的 $if ~ k = n ~ then$
> $$P\left(n, n\right) = n! = \prod_{i = 1}^{n} i, 1 \le n$$
>
> 当 $n$ 很大时可以根据 **斯特林公式**,$\pi$ 为圆周率，$e$ 为自然对数。
> $$n! \approx \sqrt{2 \pi n} \left(\frac{n}{e}\right)^{n}$$
>
> **组合**
> $$C\left(n, k\right) = C_{n}^{k} = \binom{n}{k} = \binom{n}{n-k} = \frac{P\left(n, k\right)}{P\left(k, k\right)} = \frac{n!}{\left(n - k!\right) k!}, 1 \le n, 0 \le k \le n$$

**多重集的排列问题**

1. 拆分原理：分步选位置

构造满足条件的序列，可拆解为$N$步选择“元素$i$的位置”，每步的选法对应一个组合数：

- **第1步**：从$T$个位置中选$C_1$个放元素$1$，选法数为$\binom{T}{C_1}$（组合数，含义是“从$T$个中选$C_1$个的方案数”）;
- **第2步**：从剩余$T - C_1$个位置中选$C_2$个放元素$2$，选法数为$\binom{T - C_1}{C_2}$；
- **第3步**：从剩余$T - C_1 - C_2$个位置中选$C_3$个放元素$3$，选法数为$\binom{T - C_1 - C_2}{C_3}$；
- ...
- **第$N$步**：剩余$C_N$个位置全放元素$N$，选法数为$\binom{C_N}{C_N} = 1$。

2. 乘积化简：等价于多项式系数

将每步的选法数相乘，化简后恰好等于多项式系数：

$$
\begin{align*}
&\binom{T}{C_1} \times \binom{T - C_1}{C_2} \times \binom{T - C_1 - C_2}{C_3} \times \dots \times \binom{C_N}{C_N} \\
=& \frac{T!}{C_1! \cdot (T - C_1)!} \times \frac{(T - C_1)!}{C_2! \cdot (T - C_1 - C_2)!} \times \dots \times \frac{C_N!}{C_N! \cdot 0!} \\
=& \frac{T!}{C_1! \cdot C_2! \cdot \dots \cdot C_N!}
\end{align*}
$$

（中间项$(T - C_1)!$、$(T - C_1 - C_2)!$等相互抵消，最终得到多项式系数）。

故，如果 $n$ 个元素中有部分元素重复，将 $n$ 元素进行全排列能够得到的不同排列方式总数为
$$\binom{n}{m_{1}, m_{2}, \cdots m_{k}} = \frac{n!}{m_{1} m_{2}! \cdots m_{k}!}, m_{1} + m_{2} + \cdots + m_{k} = n$$

$m_{i}$ 表示第 $i$ 个不同元素的重复次数。

例如，将字符串"$aabdeef$"进行全排列，能够得到的不同字符串排列种数为
$$P = \frac{7!}{2! \cdot 1! \cdot 1! \cdot 2! \cdot 1!} = 1260$$

为了便于计算组合数，还可以使用以下递推公式
$$C(n, k) = C(n - 1, k) + C(n - 1, k - 1)$$

```c++
long long Cnk[41][41] = {0};
for(int i = 0; i <= 40; i ++){
    Cnk[i][0] = Cnk[i][i] = 1;
    for(int j = 1; j < i; j ++){
        Cnk[i][j] = Cnk[i - 1][j] + Cnk[i - 1][j - 1];
    }
}
```

**注：** 因为阶乘增长得很快，在某些情况下，虽然给定组合数的最终结果在 $64$ 位整数的表示范围内，但直接按照定义计算，*中间结果*会超出 $64$ 整数的表示范围，此时需要采用一些特殊的技巧。

- 将阶乘的各个数进行素因子分解，消去分子和分母共同的**素因子**使中间数值变小。
- 计算式中分子各个乘数和分母各个乘数间的**最大公约数**来使中间结果变小。

```c++
const int N = 1e6 + 10;
const ll MOD = 1e9 + 7;

ll fac[N];
ll inv_fac[N];

ll qpow(ll a, ll n){
    ll ans = 1;
    while(n){
        if(n & 1){
            ans = ans * a % MOD;
        }
        a = a * a % MOD;
        n >>= 1;
    }
    return ans;
}

void init(){
    fac[0] = 1;
    for(int i = 1; i < N; i ++){
        fac[i] = fac[i - 1] * i % MOD;
    }
    inv_fac[N - 1] = qpow(fac[N - 1], MOD - 2);
    for(int i = N - 2; i >= 0; i --){
        inv_fac[i] = inv_fac[i + 1] * (i + 1) % MOD;
    }
}

ll C(ll n, ll k){
    if(n < 0 || k < 0 || n < k) return 0;
    return fac[n] * inv_fac[k] % MOD * inv_fac[n - k] % MOD;
}
```



## 几何

### 极角排序

### 数学定义与几何意义

对于二维平面上的两个向量 $\vec{a} = (x_1, y_1)$ 和 $\vec{b} = (x_2, y_2)$，它们的叉积被定义为一个**标量**（即一个实数）：
$$
\vec{a} \times \vec{b} = x_1 \cdot y_2 - y_1 \cdot x_2
$$
**几何意义**：这个标量的绝对值等于以 $\vec{a}$ 和 $\vec{b}$ 为邻边构成的平行四边形的**有向面积**。

### 符号与方向判断

叉积的符号直接告诉我们 $\vec{b}$ 相对于 $\vec{a}$ 的**旋转方向**。

假设你站在原点，面朝向量 $\vec{a}$ 的方向：
- **若 $\vec{a} \times \vec{b} > 0$**（叉积为正）：$\vec{b}$ 在你的**左手边**（逆时针方向）。
- **若 $\vec{a} \times \vec{b} < 0$**（叉积为负）：$\vec{b}$ 在你的**右手边**（顺时针方向）。
- **若 $\vec{a} \times \vec{b} = 0$**：两个向量**共线**（在同一条直线上），可能同向也可能反向。

### 模版

```c++
template <typename T>
struct point{
    T x, y;
    point() : x(0), y(0){}
    point(T _x, T _y) : x(_x), y(_y){}
    // 叉乘
    T cross(const point<T> &b)const{
        return x * b.y - y * b.x;
    }
    T cross(const point<T> &a, const point<T> &b){
        return a.x * b.y - a.y * b.x;                                               
    }
    // 逆时针
    bool operator<(const point<T> &nxt)const{
        // 是否在 x 轴上方
        bool a = y > 0 || y == 0 && x > 0;
        bool b = nxt.y > 0 || nxt.y == 0 && nxt.x > 0;
        if(a ^ b) return a; // 不同区域
        return cross(nxt) > 0;
    }
    // 到达 点o 的距离的平方 
    T dist2(const point<T> &o)const{
        return (x - o.x) * (x - o.x) + (y - o.y) * (y - o.y);
    }
};
```

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Point {
    ll x, y;
    int quad() const {   // 获取象限（逆时针顺序：1→2→3→4）
        if (x > 0 && y >= 0) return 1;
        if (x <= 0 && y > 0) return 2;
        if (x < 0 && y <= 0) return 3;
        if (x >= 0 && y < 0) return 4;
        return 0; // 原点
    }
    ll cross(const Point& b) const { return x * b.y - y * b.x; }
};

// 逆时针极角排序比较器
bool cmp_polar(const Point& a, const Point& b) {
    if (a.quad() != b.quad()) return a.quad() < b.quad();
    ll c = a.cross(b);
    if (c != 0) return c > 0;          // 逆时针方向
    return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y; // 共线按距离
}

// 使用示例
vector<Point> pts = { {1,1}, {-1,2}, {-2,-1}, {2,-1} };
sort(pts.begin(), pts.end(), cmp_polar);
```

如果基点不是原点

```c++
Point base = {1, 1};   // 假设基准点
for (auto& p : pts) { p.x -= base.x; p.y -= base.y; }
sort(pts.begin(), pts.end(), cmp_polar);
for (auto& p : pts) { p.x += base.x; p.y += base.y; }
```



## 三角形面积

```c++
double get_area(ll x1, ll y1, ll x2, ll y2){
    return 0.5 * (double)abs(x1 * y2 - y1 * x2);
}

double get_area(ll a, ll b, ll c){
    double s = 0.5 * (double)(a + b + c);
    ll val = s * (s - a) * (s - b) * (s - c);
    return sqrt(max(0ll, val));
}
```

## 多边形求面积

```c++
struct Point {
    ll x, y;
};

// 鞋带公式求多边形面积，返回面积的两倍（避免浮点运算），实际面积需除以2.0
ll area2(const vector<Point>& poly) {
    int n = poly.size();
    if (n < 3) return 0; // 不是多边形
    ll sum = 0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        sum += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return abs(sum);
}

// 若需返回浮点数面积
double area(const vector<Point>& poly) {
    return area2(poly) / 2.0;
}
```

## 格雷码

> 对于长度为 $2^{n}$ 的格雷码序列(包含 $0, 1, \cdots, 2^{n} - 1$ 所有的整数)，*任意相邻两个数的二进制表示仅 $1$ 位不同*，$且序列第一个和最后一个数的二进制也仅 $1$ 位不同。 

### 公式

$$gray(i) = i \oplus (i >> 1) $$

### 应用

- 构造 $0 \sim 2^{n} - 1$ 的排列，使相邻异或和最小。

## 数学期望

对于一个 *离散型随机变量* $X$，如果它可能得取值为 $\{ x_{i} \}$，对于每个取值对应的概率为 $\{ p_{i} \}$，那么 $X$ 的数学期望 $E[X]$ 为:
$$
E(X) = \sum_{i = 1}^{n} x_{i} \cdot p_{i}
$$
**期望的线性性质**:
$$
E[X + Y] = E[X] + E[Y]
$$

$$
E[a \cdot X] = a \cdot E[X]
$$

## 高精度

### 高精度加法 

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

## 约瑟夫问题

> $n$ 个人围成一圈，从第一个人开始报数，每数到第 $k$ 个人就将他处决，然后从下一个人继续报数。求最后存活下来的人的**初始位置编号**。

假设人的编号从 $0$ 到 $n-1$（若题目要求从 $1$ 开始，最后结果 $+1$ 即可），记 $f(n,k)$ 表示 $n$ 个人、每 $k$ 个一杀时最后存活者的编号。

**递推公式:**
$$
f(1,k) = 0
$$

$$
f(n,k) = (f(n-1,k) + k) \bmod n, \quad n \ge 2
$$

通过这个 $O(n)$ 的递推，可以求出任意 $n$ 和 $k$ 的结果。

### 高精度减法

```c++
// 返回 true 表示 a >= b，false 表示 a < b
bool greaterOrEqual(string a, string b) {
    if (a.size() != b.size()) return a.size() > b.size();
    return a >= b;   // 长度相同时，字典序比较等价于数值比较
}
string subtract(string a, string b) {
    // 1. 处理符号：若 a < b，交换并标记为负
    bool negative = false;
    if (!greaterOrEqual(a, b)) {
        swap(a, b);
        negative = true;
    }

    // 2. 核心减法逻辑（假设 a >= b）
    string s = "";
    int borrow = 0;
    for (int i = a.size() - 1, j = b.size() - 1; i >= 0 || j >= 0; i--, j--) {
        int x = (i >= 0) ? (a[i] - '0') : 0;
        int y = (j >= 0) ? (b[j] - '0') : 0;
        x -= borrow;
        if (x < y) {
            x += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        s += to_string(x - y);
    }

    // 3. 反转并去除前导零
    reverse(s.begin(), s.end());
    size_t pos = s.find_first_not_of('0');
    if (pos == string::npos) return "0";   // 结果为 0 时不加负号
    s = s.substr(pos);

    // 4. 添加负号（如果需要）
    if (negative) s = "-" + s;
    return s;
}
```



# 图论

## 链式前向星

```c++
// 最大节点数
const int MAXN = 10 + 5;
// 最大边数
const int MAXM = 20 + 5;
int head[MAXN]; // idx:节点号   val:边的编号id
// 有向图
int nxt[MAXM]; // idx:边的编号id   val:下一条同起点边的编号
// 无向图
// int next[MAXM << 1]; 
int to[MAXM];   // idx:边的编号id   val:节点号
// int w[MAXM];    // idx:边的编号id   val:边的权重
int id = 1; // 边的编号
void add_edge(int _u, int _v){    // _u->_v
    nxt[id] = head[_u];
    to[id] = _v;
    head[_u] = id ++;
}
// void add_edge(int _u, int _v, int _w){ // _u->_v, weight = _w
//     nxt[id] = head[_u];
//     to[id] = _v;
//     w[id] = _w;
//     head[_u] = id ++;
// }

/*
// 遍历 u 的所有后继节点
for(int i = head[u]; i; i = nxt[i]){
    int v = to[i]; // u->v
    // int val = weight[i] // u->v的权重
}
*/

// memset(head, 0, sizeof(head));
```

## Dijkstra

```c++
# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'
const ll INF = 0x3f3f3f3f3f3f3f3fLL;
const int N = 3e5 + 2;
struct edge{
    int from, to; // 起点、终点; 起点 from 没有用到，e[i] 的 i 就是 from
    ll w;   // 权值
    edge(int _from, int _to, ll _w): from(_from), to(_to), w(_w){}
};
struct node{
    int id; // 节点
    ll dis; // 节点到起点的距离
    node(int _id, ll _dis): id(_id), dis(_dis){}
    bool operator<(const node& a)const{
        return dis > a.dis;
    }
};
int n, m, s;
vector<int> pre(N, -1); // 记录前驱节点，用于生成路径
vector<vector<edge> > mat(N, vector<edge>{}); // 存图
vector<ll> dis(N, 0); // 记录所有节点到起点的距离；
void print_path(int s, int t){ // 输出从 s 到 t 的最短路径
    if(s == t){ // 输出起点
        cout << s << " ";
        return;
    }  
    print_path(s, pre[t]); // 输出前一个点
    cout << t << " "; // 后输出当前点。最后输出终点；
}
void dijkstra(){
    vector<bool> done(N, false); // true表示节点i的最短路径已经找到
    for(int i = 1; i <= n; i ++){ // init
        dis[i] = INF;
        done[i] = false;
    }
    dis[s] = 0; // 起点到自己的距离为0
    priority_queue<node> pq; // 存储节点信息
    pq.emplace(node(s, dis[s])); // 起点入队
    while(!pq.empty()){ 
        node cur = pq.top(); pq.pop(); // pop 出与起点 s 距离最小的节点 cur
        if(done[cur.id]) continue; // 丢弃已经找到最短路径的节点
        done[cur.id] = true;
        for(int i = 0; i < mat[cur.id].size(); i ++){ // 检查节点 cur 的所有邻居节点
            edge nxt = mat[cur.id][i]; // cur.id 的第 i 个邻居节点是 nxt.to
            if(done[nxt.to]) continue; // 丢弃已经找到最短路径的邻居节点
            if(dis[nxt.to] > nxt.w + cur.dis){
                dis[nxt.to] = nxt.w + cur.dis;
                pq.emplace(node(nxt.to, dis[nxt.to]));  // 扩展新的邻居节点，放到优先队列中
                pre[nxt.to] = cur.id;   // 如果有需要，就记录路径
            }
        }
    }
    //print_path(s, n); cout << endl; // 输出路径; 起点为 1，终点 n;
}

void solve(void){
    cin >> n >> m;
    s = 1; // 起点
    for(int i = 1; i <= n; i ++) mat[i].clear();
    while(m --){
        int a, b, w; cin >> a >> b >> w;
        mat[a].emplace_back(edge(a, b, w));
        // mat[b].emplace_back(edge(b, a, w)); // 双向
    }
    dijkstra();
    for(int i = 1; i <= n; i ++){
        if(dis[i] >= INF){
            cout << "-1 ";
        }else{
            cout << dis[i] << " ";
        }
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

## Floyd

```c++
const int INF = 0x13131313;

void solve(void){
    int n, m; // n 个节点，m 条边
    cin >> n >> m;
    vector<vector<int> > mat(n + 1, vector<int> (m, INF));  // 图
    vector<vector<int> > path(n + 1, vector<int> (m, -1));  // 路径
    for(int i = 0; i < m; i ++){
        int u, v, w; cin >> u >> v >> w;
        mat[u][v] = w;
        path[u][v] = u;
    }
    vector<vector<int> > dis(mat.begin(), mat.end());   // 最短路径
    
    for(int k = 1; k <= n; k ++){   // 中间节点
        for(int i = 1; i <= n; i ++){   // 左端节点
            for(int j = 1; j <= n; j ++){   // 右端节点
                if(dis[i][k] + dis[k][j] < dis[i][j]){
                    dis[i][j] = dis[i][k] + dis[k][j];
                    path[i][j] = path[k][j];
                }
            }
        }
    }
    
}
```

## 并查集

```c++
int n, m; 
vector<int> arr;

void init(){
    for(int i = 0; i <= n; i ++) arr[i] = i;
}
int getFa(int a){ 
    if(a == arr[a]) return a; // 如果该节点指向自己
    //return getFa(arr[a]);
    return arr[a] = getFa(arr[a]);// 建森林
}
void Union(int x, int y){
    int X = getFa(x), Y = getFa(y); // 获取父节点
    if(X == Y) return; // 两节点在同一集合中
    arr[X] = Y; // 将两节点进行连接
}

void solve(){
    cin >> n >> m;
    arr.resize(n + 1);
    init();
    int a, b;
    for(int _ = 0; _ < m; _ ++){
        cin >> a >> b;
        Union(a, b);
    }
    //······
}
```

## 最小生成树

### Kruskal(克鲁斯卡尔)算法

> 时间复杂度: $O(mlogm)$

### 算法步骤

1. 创建一个空的最小生成树 $tree$。
2. 将图中所有边按照权重从小到大排序。
3. 从权重最小的边开始，判断其连接的连个节点是否在 $tree$ 中，若不在则加入。
4. 重复步骤 $3$ 直到所有点都包含在 $tree$ 中。

### 代码实现

```c++
struct edge{
    int a, b, val;  // a 节点，b 节点， 边的权重
    edge():a(0), b(0), val(0){}
    edge(int _a, int _b, int _val): a(_a), b(_b), val(_val){}
    bool operator<(const edge &nxt)const{
        return val < nxt.val;
    }
};

int n, m;   // 节点数，边数
vector<edge> Edge;  // 边集
vector<int> tree;    // 并查集，最小生成树 MST

void init(){
    cin >> n >> m;
    Edge.resize(m);
    tree.resize(n + 1, 0);
    for(int i = 0; i < m; i ++){    // 初始化边
        cin >> Edge[i].a >> Edge[i].b >> Edge[i].val;
    }
    for(int i = 1; i <= n; i ++){   // 初始化并查集
        tree[i] = i;
    }
}

int getFa(int x){   // 获得其父亲节点(建森林，路径压缩)
    if(tree[x] == x) return x;
    return tree[x] = getFa(tree[x]);
}

void Union(int x, int y){ // 将连个节点连接
    int X = getFa(x), Y = getFa(y);
    if(X == Y) return;
    tree[X] = Y;
}

int kruskal(){
    sort(Edge.begin(), Edge.end());
    int res = 0;    // 最小生成树的权重
    int cnt = 0;    // 已连接边数
    for(edge &it : Edge){
        int A = getFa(it.a);
        int B = getFa(it.b);
        if(A == B) continue; // 两个节点在同一个连通分量中
        cnt ++;
        res += it.val;
        Union(it.a, it.b);
    }
    if(cnt < n - 1){ // 无法构成最小生成树
        return INT_MAX;
    }
    return res;
}

void solve(){
    init();
    int res = kruskal();
    if(res == INT_MAX){
        cout << "None" << endl;
        return;
    }
    cout << res << endl;
}
```

## 强连通分量(Tarjan算法)

### 基本概念

- 边分为三种类型: 树边、回边、弃边。
- `dfn[u]` 节点 $u$ 获得的 $dfn$ 序号。
- `low[u]` 节点 $u$ 及其子树上的点，最多走一条回边，能到达的树的最上方点的 $dfn$ 序号。
- `belong[n]` 节点 $u$ 分配的强连通分量序号，如果 `belong[u] == 0`，则表示没有分配。

### 算法流程

1. 初次到达 $u$ 时，`dfn[u] = low[u] = 分配的 dfn 序号`，然后节点 u 进栈
2. 遍历 $u$ 的每个儿子 $v$，讨论 $u$ 和 $v$ 之间的边是什么类型，如果是弃边，什么也不做
3. 如果是树边，先递归计算 `low[v]`，然后 `low[u] = min(low[u], low[v])`
4. 如果是回边，直接 `low[u] = min(low[u], dfn[v])`
5. 遍历结束后，判断 `dfn[u]` 是否等于 `low[u]`，如果不等，代表节点 $u$ 不能扎起口袋
6. 如果相等，代表节点 $u$ 能扎起口袋，栈弹出节点，直到 $u$ 出现时停止，这批节点属于一个强连通分量
7. 考察途中每个点，一旦发现 `dfn[i] == 0`, 就从 $i$ 节点开始，执行一遍 $Tarjan$ 算法
8. 最终得到图中全部强连通分量，时间复杂度 $O(n + m)$，$n$ 为节点数，$m$ 为边数

 模版

```c++
const int N = 1e4 + 10;
const int M = 1e5 + 10;

int head[N] = {0};
int nxt[M] = {0};
int to[M] = {0};
int cnt_e = 1;
int n, m;

int dfn[N] = {0}; // dfn[u] := 节点 u 获得的 dfn 序号
int low[N] = {0}; // low[u] := 节点 u 及其子树上的点，最多走一次回边，能到达的树的最上方点的 dfn 序号
int belong[N] = {0}; // belong[u] := 节点 u 的强连通分量序号
int cnt_d = 0; // dfn 序号
int cnt_scc = 0; // 强连通分量个数

int sta[N] = {0}; // 栈
int top = 0; // 栈顶


void add_edge(int u, int v){
    nxt[cnt_e] = head[u];
    to[cnt_e] = v;
    head[u] = cnt_e ++;
}

void tarjan(int u){
    dfn[u] = low[u] = ++cnt_d;
    sta[++ top] = u;
    for(int e = head[u]; e > 0 ; e = nxt[e]){
        int v = to[e];
        if(!dfn[v]){
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }else{
            if(!belong[v]){ // 是回边
                low[u] = min(low[u], dfn[v]);
            }
        }
    }
    if(dfn[u] == low[u]){ // 是强连通分量
        cnt_scc ++;
        int pop;
        do{
            pop = sta[top --];
            belong[pop] = cnt_scc;
        }while(pop != u);
    }
}

void solve(void){
    cin >> n >> m;
    for(int i = 1; i <= m; i ++){
        int u, v; cin >> u >> v;
        add_edge(u, v);
    }
    for(int i = 1; i <= n; i ++){
        if(!dfn[i]){
            tarjan(i);
        }
    }
    vector<vector<int> > ans(cnt_scc);
    for(int i = 1; i <= n; i ++){
        ans[belong[i] - 1].emplace_back(i);
    }
    sort(ans.begin(), ans.end());
    cout << cnt_scc << endl;
    for(auto arr : ans){
        for(int i = 0; i < arr.size(); i ++){
            if(i > 0) cout << " ";
            cout << arr[i];
        }cout << endl;
    }
}

signed main(void){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}

```



# 子序列自动机

## 基本概念

> **子序列自动机** 是一种高效处理 *子序列匹配* 问题的数据结构。

**核心思想：** 预处理出从每个位置 $i$ 开始(包括 $i$ )，每个字符 $c$ 下一次出现的位置。

**定义：**

- `n`: 字符串 $s$ 的长度。
- $\sum$ : 字符集大小。

- `nxt[i][c]`: 表示从位置 $i ~ (0 \le i \le n)$ 开始，字符 $c$ 第一次出现的位置。若不存在，则设为 $n$ (或者 $n + 1$ )。

## 构建方法

### 二维数组法(小字符集)

```c++
string s; cin >> s;
int n = s.size();
vector<vector<int> > nxt(n + 1, vector<int> (26, n));
for(int i = n - 1; i >= 0; i --){
    nxt[i] = nxt[i + 1];
    nxt[i][s[i] - 'a'] = i;
}
```

**时间复杂度:** $O(n \cdot \sum)$

### 二分查找法(大字符集)

```c++
vector<int> pos[N]; // N: 字符集大小
for(int i = 0; i < n; i ++){
    pos[arr[i]].emplace_back(i);
}
// 查询字符 c 在位置 cur 之后的下一个位置
auto it = upper_bound(pos[c].begin(), pos[c].end(), cur);
if(it == pos[c].end()){
    // 不存在
}else{
    // 下一个位置 = *it;
}
```

**时间复杂度:** 构建 $O(n)$，单次查询 $O(log~n)$

## 主要应用

### 判断是否为子序列

> 给出模式串 $T$，判断是否为 $S$ 的子序列。 

这个给出针对 *小字符集* 的判断方法。

```c++
int cur = 0;
for(char ch : T){
    cur = nxt[cur][ch - 'a'];
    if(cur == n) return false;
}
return true;
```

### 计算不同的子序列

> [不同的子序列 II](https://leetcode.cn/problems/distinct-subsequences-ii/description/ "不同的子序列 II") : 给定一个字符串 $s$，计算 $s$ 的 *不同非空子序列* 的个数(答案对 `1e9 + 7` 取余)。

#### 子序列自动机+DP

**定义:**

- `dp[i]`: 从下标 $i$ 开始到字符串末尾，能够组成的 *所有不同子序列的数量* (包括空串)。

**状态转移方程:** 
$$
dp[i] = 1 + \sum_{c = 0}^{25} dp\left[nxt[i][c] + 1\right]
$$

- `1`: 表示当前位置留空。
- $\sum_{c = 0}^{25} dp\left[nxt[i][c] + 1\right]$: 枚举当前位置需要放置的每个字符 $c$ 。
    - `nxt[i][c]`: 表示从 $i$ 位置开始下一个字符 $c$ 的位置。
    - `nxt[i][c] + 1`: 确定字符 $c$ 后，字符 $c$ 的下一个位置。
    - `dp[nxt[i][c] + 1]`: 字符 $c$ 后面可以跟的不同子序列的数量。

**代码:**

```c++
class Solution {
public:
    int distinctSubseqII(string s) {
        int n = s.size();
        int mod = 1e9 + 7;
        // 初始化 子序列自动机
        vector<vector<int> > nxt(n + 1, vector<int> (26, n)); 
        for(int i = n - 1; i >= 0; i --){
            nxt[i] = nxt[i + 1];
            nxt[i][s[i] - 'a'] = i;
        }
        // dp[i] := 从下标 i 开始到字符串末尾，能够组成的 所有不同子序列 的数量(包括空串)
        vector<int> dp(n + 1, 0);
        dp[n] = 1; // 空串的不同子序列数量也为 1
        for(int i = n - 1; i >= 0; i --){
            dp[i] ++; // 当前位置字符为空
            for(int c = 0; c < 26; c ++){
                if(nxt[i][c] != n){ // 避免 i 位置及以后不存在字符 c
                    dp[i] = ((long long)dp[i] + dp[nxt[i][c] + 1]) % mod;
                }
            }
        }
        return (dp[0] - 1 + mod) % mod; // 排除 dp[0] - 1 取负
    }
};
```

**时间复杂度:** $O(n \cdot \sum)$

#### 线性DP

> 这里更加推荐使用 线性DP 来解决这题。

**定义:**

- `last[c]`: 字符 c 最后出现的位置。

- `dp[i]`: $s[0, i]$ 子串中不同子序列的数量。

**状态转移方程**:
$$
dp[i] = (dp[i - 1] \times 2 - dp[last[c] - 1] + MOD) ~ \% ~ MOD
$$

- `dp[i - 1] * 2`: $i$ 位置有 选 和 不选 两种状态。
- `- dp[last[c] - 1]`: 之前出现过字符 $c$ ，会重复考虑 $s\left[0,~ last[c] - 1\right]$ 部分的 不同子序列个数。

**代码:**

```c++
class Solution {
public:
    int distinctSubseqII(string s) {
        int n = s.size();
        int mod = 1e9 + 7;
        s = " " + s; // 统一起始索引从 1 开始
        unordered_map<char, int> last; // last[c]:= 字符 c 最后出现的位置
        // dp[i] := s[0, i]子串中不同子序列的数量 
        vector<int> dp(n + 1, 0);
        dp[0] = 1; // 空串的不同子序列数量为 1
        for(int i = 1; i <= n; i ++){
            dp[i] = dp[i - 1] * 2 % mod;
            if(last.count(s[i])){ // 减去重复部分
                dp[i] = (dp[i] - dp[last[s[i]] - 1] + mod) % mod;
            }
            last[s[i]] = i; // 更新字符最后出现的位置
        }
        return (dp[n] - 1 + mod) % mod;
    }
};
```

**时间复杂度:** $O(n)$

### 字典序第 K 小的子序列

> [[R23E]第k小子序列](https://bs.daimayuan.top/p/140 "[R23E]第k小子序列"): 给定给定一个长度为 $n$ 且仅由小写字母构成的字符串，求该串种字典序第 $k$ 小的本质不同的非空子序列。

**定义:**

- `nxt[i][j]`: 表示从 $i$ 位置开始下一个字符 $c$ 的位置。
- `dp[i]`: $s[0, i]$ 子串中不同子序列的数量。

**注:** 这两个的具体定义前面有讲，这里就不再过多赘述。

**代码:**

```c++
const ll K = 1e18;

void solve(){
    ll n, k;
    cin >> n >> k;
    string s; cin >> s;
    s = " " + s;
    // 初始化 子序列自动机
    vector<vector<ll> > nxt(n + 2, vector<ll> (26, n + 1));
    for(int i = n; i >= 1; i --){
        nxt[i] = nxt[i + 1];
        nxt[i][s[i] - 'a'] = i;
    }
    // dp[i] := s[i, n]子串中 不同子序列个数
    vector<ll> dp(n + 2, 0);
    dp[n + 1] = 1; // 空串的不同子序列数量也为 1
    for(ll i = n; i >= 1; i --){
        dp[i] = 1; // 当前位置为空
        for(ll c = 0; c < 26; c ++){ // 枚举该位置可能得所有字符
            if(nxt[i][c] != n + 1){ // nxt[i][c] == n + 1 代表 s[i, n] 不存在字符 c
                ll add = dp[nxt[i][c] + 1]; 
                if(dp[i] > K - add){ // 防止溢出
                    dp[i] = K + 1;
                } else {
                    dp[i] += add;
                }
            }
        }
    }
    string ans = "";
    ll pre = 1; // 字符的索引基点
    // 贪心的枚举每个可能字符
    while(k > 0){
        for(ll c = 0; c < 26; c ++){ // 从小到大枚举每个字符
            ll pos = nxt[pre][c]; 
            if(pos == n + 1) continue; // 不存在该字符
            ll cnt = dp[pos + 1];
            if(k > cnt){ // 使用字符 c 的所有子序列可能数比 k 要小，
                k -= cnt;
            }else{
                ans.push_back('a' + c);
                k --; // 到该位置后截断的可能
                pre = pos + 1; // 更新 基点
                break;
            }
        }
    }
    cout << ans << endl;
}

signed main(void){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

**时间复杂度:** $O(n \cdot \sum)$ 

# 动态规划

## 最长递增子序列问题

> 给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

### 暴力

**时间复杂度**: $O(n^2)$

```c++
int lengthOfLIS(vector<int>& nums) { 
	int n = nums.size(); 
	vector<int> dp(n, 1); 
	int ans = 1; 
	for(int i = 1; i < n; i ++){ 
		for(int j = 0; j < i; j ++){ 
			if(nums[j] < nums[i]) dp[i] = max(dp[i], dp[j] + 1); 
		} 
		ans = max(ans, dp[i]); 
	} return ans; 
}
```

### 二分优化

**时间复杂度**: $O(n\log n)$

```c++
int lengthOfLIS(vector<int>& nums) {
	int n = nums.size();
	// 长度为 i + 1 的递增子序列的最小结尾
	vector<int> ends(n, -1);
	int len = 0;
	for(int i = 0; i < n; i ++){
		int inx = lower_bound(ends.begin(), ends.begin() + len, nums[i]) - ends.begin();
		if(inx == len){
			ends[len ++] = nums[i];
		}else{
			ends[inx] = nums[i]; 
		}
	}
	return len;        
}
```

## 背包问题

### 01背包

`t`: 表示背包的容量。
`n`: 货物数量，每个货物可选择 1 次。
`costs[i]`: 第`i`个货物的体积。
`values[i]`: 第`i`个货物的价值。

```c++
void solve(){
	int t, n;
	// cin>> t >> n;
	vector<int> costs(n), values(n);
	for(int i = 0; i < n; i ++) cin >> costs[i];
	for(int i = 0; i < n; i ++) cin >> values[i];
	// dp[i][j] := 考虑前 i 个物品，在容量为 j 的时候能拿的最大价值。
	vector<vector<int> > dp(n + 1, vector<int> (t + 1, 0));
	for(int i = 1; i <= n; i ++){
		for(int j = 0; j <= t; j ++){
			dp[i][j] = dp[i - 1][j];
			if(costs[i - 1] <= j) dp[i][j] = max(dp[i][j], dp[i - 1][j - costs[i - 1]] + values[i - 1]);
		}
	}
	cout << dp[n][t] << endl;
}
// 空间压缩
void solve(){
	int t, n;
	// cin>> t >> n;
	vector<int> costs(n), values(n);
	for(int i = 0; i < n; i ++) cin >> costs[i];
	for(int i = 0; i < n; i ++) cin >> values[i];
	// dp[i] := 容量为 i 的背包，能拿的最大价值  
    vector<int> dp(t + 1, 0);
    for(int i = 0; i < n; i ++){
        for(int j = t; j >= costs[i]; j --){
            dp[j] = max(dp[j], dp[j - costs[i]] + values[i]);
        }
    }
    cout << dp[t] << endl;
}
```

### 多重背包

#### 暴力

`t`: 物品数量
`m`: 背包容量
`vs[i]`: 物品 i 的价值
`ws[i]`: 物品 i 的重量
`ns[i]`: 物品 i 的数量

```c++
void solve(void){
    int t, m; cin >> t >> m;
    vector<int> vs(t + 1, 0);
    vector<int> ws(t + 1, 0);
    vector<int> ns(t + 1, 0);
    for(int i = 1; i <= t; i ++){
        cin >> vs[i] >> ws[i] >> ns[i];
    }
    // dp[i][j] := 考虑到第 i 个宝物，背包容量为 j 时，能够获得的最大价值
    // vector<vector<int> > dp(t + 1, vector<int> (m + 1, 0));
    vector<int> dp(m + 1, 0);
    for(int i = 1; i <= t; i ++){
        for(int j = m; j >= 0; j --){
            for(int n = 1; n <= ns[i] && n * ws[i] <= j; n ++){
                dp[j] = max(dp[j], dp[j - n * ws[i]] + n * vs[i]);
            }
        }
    }
    cout << dp[m] << endl;
}
```

#### 二进制分组

> 通过 二进制分组 将多重背包转化为 01背包。
> **时间复杂度:** $O\left(m \sum_{i = 1}^{t}\log(cnt(i))\right)$

```c++
void solve(void){
    int t, m; cin >> t >> m;
    vector<int> vs;
    vector<int> ws;
    for(int i = 0; i < t; i ++){
        int v, w, n;
        cin >> v >> w >> n;
        for(int k = 1; n >= k; k <<= 1){
            vs.emplace_back(k * v);
            ws.emplace_back(k * w);
            n -= k;
        }
        if(n > 0){
            vs.emplace_back(n * v);
            ws.emplace_back(n * w);
        }
    }
    int n = vs.size();
    vector<int> dp(m + 1, 0);
    for(int i = 0; i < n; i ++){
        for(int j = m; j >= ws[i]; j --){
            dp[j] = max(dp[j], dp[j - ws[i]] + vs[i]); 
        }
    }
    cout << dp[m] << endl;
}
```

### 分组背包

```c++
void solve(){
	int teams; // 分组数量
	int m; // 背包容量
	vector<vector<int> > dp(teams + 1, vector<int> (m + 1, 0));
	for(int i = 1; i <= teams; i ++){
		for(int j = 0; j <= m; j ++){
			dp[i][j] = dp[i - 1][j];
			for(int k : team[i]){
				if(j - arr[k] >= 0){
					dp[i][j] = max(dp[i][j], dp[i - 1][j - arr[k]] + val[k]);
				}
			}
		}
	}
	cout << dp[teams][m] << endl;
}
```

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

# 分块

## 数组分块

### 单点修改

> [Give Away](https://www.luogu.com.cn/problem/SP18185)
> 给定一个长度为 $n$ 的数组 `arr`，接下来 $m$ 条操作，每条操作是如下两种类型种的一种:
> `0 a b c`: 打印 `arr[a..b]` 范围上 $\ge c$ 的数字个数
> `1 a b`: 把 `arr[a]` 的值改成 $b$。
> $1 \le n \le 5e5$
> $1 \le m \le 1e5$

```c++
const int N = 5e5 + 10;
const int B = 1e3 + 10;

int n, m; // n := 数组长度  m := 操作条数
int blen, bnum; // blen:= 分组长度  bnum := 分组个数    
int arr[N];  // 原数组
int sortv[N];   // 各分组内部排序
int bi[N]; // bi[i] := i 索引对应的分组编号
int bl[B], br[B]; // 分组对应的原数组左右边界


void build(){
    blen = sqrt(n);
    bnum = (n + blen - 1) / blen;
    for(int i = 1; i <= n; i ++){
        bi[i] = (i - 1) / blen + 1;
    }
    for(int i = 1; i <= bnum; i ++){
        bl[i] = (i - 1) * blen + 1;
        br[i] = min(i * blen, n);
    }
    for(int i = 1; i <= n; i ++){
        sortv[i] = arr[i];
    }
    for(int i = 1; i <= bnum; i ++){
        sort(sortv + bl[i], sortv + br[i] + 1);
    }
}

void update(int inx, int val){
    int lf = bl[bi[inx]];
    int rt = br[bi[inx]];
    arr[inx] = val;
    sort(sortv + lf, sortv + rt + 1);
}
// 第 inx 组中 >= val 的元素个数
int get_cnt(int inx, int val){
    int lf = bl[inx], rt = br[inx];
    int ans = 0, mid;
    while(lf + 1 < rt){
        mid = lf + ((rt - lf) >> 1);
        if(sortv[mid] >= val){
            ans += rt - mid;
            rt = mid;
        }else{
            lf = mid;
        }
    }
    if(sortv[lf] >= val) ans ++;
    if(sortv[rt] >= val) ans ++;
    return ans;
}

int query(int lf, int rt, int val){ // >= val 个数
    int ans = 0;
    if(bi[lf] == bi[rt]){
        for(int i = lf; i <= rt; i ++){
            if(arr[i] >= val) ans ++;
        }
    }else{
        for(int i = lf; i <= br[bi[lf]]; i ++){
            if(arr[i] >= val) ans ++;
        }
        for(int i = bl[bi[rt]]; i <= rt; i ++){
            if(arr[i] >= val) ans ++;
        }
        for(int i = bi[lf] + 1; i < bi[rt]; i ++){
            ans += get_cnt(i, val);
        }
    }
    return ans;
}

void solve(void){
    cin >> n;
    for(int i = 1; i <= n; i ++){
        cin >> arr[i];
    }
    build();
    cin >> m;
    while(m --){
        int op, a, b;
        cin >> op >> a >> b;
        if(op == 0){
            int c;
            cin >> c;
            cout << query(a, b, c) << endl;
        }else{
            update(a, b);
        }
    }    
}

signed main(void){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    int t = 1;
    //cin >> t;
    while(t--){
        solve();
    }
    return 0;
}
```

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



# 字符串

## 字符串哈希

### 性质

- 输入参数的可能性无线，输出的值范围相对有限
- 输入相同样本得到相同的值（没有随机机制）
- 输入不同的样本也可能得到相同的值（哈希碰撞，可能行较小）
- 输入大量不同的样本，得到大量输出值，几乎均匀的分布在整个输出域上

### 细节

1. 理解unsigned long long类型自然溢出，计算加减乘除时，自然溢出后的状态等同于对2^64次方取模的值。
2. 字符串化成base进制的数字并让其自然溢出。
3. base可以选择一些指数比如：433、299、599、1000000007；（经典值：31、131、1313、13131、131313等）
4. 转化时让每一位的值***从1开始***，不从0开始；
5. 利用数字的比较去替代字符串比较，可以大大减少复杂度  

>注：出现哈希碰撞就换base。

### template

```c++
#define ull unsigned long long
const int N = 1e6 + 10;
ull p[N], h[N]; // p[i] = P^i, h[i] = s[1~i]的hash值
string s;
// s = " " + s; // s 从1开始计算

void init(int n){
    p[0] = 1, h[0] = 0;
    for(int i = 1; i <= n; i ++){
        p[i] = p[i - 1] * P;
        h[i] = h[i - 1] * P + s[i];
    }
}
ull get(int lf, int rt){ // 计算s[lf~rt]的hash值
    return h[rt] - h[lf - 1] * p[rt - lf + 1];
}

```

## 字符串处理

### 常用函数

#### find

- **str.find(s)**
    返回str中第一次出现s子串的位置。
- **str.find(s, pos)**
    返回str中从pos位置开始，第一次出现s子串的位置。
    注：没有找到则返回 std::string::npos (-1).

#### replace

- **str.replace(pos, len, s)**
    从 pos 位置开始将 len 长度的字符替换为 s.

### substr

- **str.substr(pos, len)**
    返回 str 从 pos 开始长度为 len 的子串。

### insert

- **str.insert(pos, s)**
    在 str 中 pos 位置添加子串 s.

## 字符串的操纵（字符串流）

> **流：** 流是一种抽象的概念，代表数据的来源和目的地。流可以是文件、控制台、内存中的字符串等。通过流，我们可以进行数据的输入（读取）和输出（写入）操作。标准 IO 库提供了一系列的流类，如 iostream、fstream 等，用于处理不同类型的流。

### 字符串流的定义

字符串流是一种特殊的流，它以字符串作为数据的来源或目的地。  
C++ 标准 IO 库提供了三个主要的字符串流类：

- **istringstream**：用于从字符串中读取数据，类似于从文件或控制台读取数据。
- **ostringstream**：用于将数据写入字符串，类似于将数据写入文件或控制台。
- **stringstream**：既可以从字符串中读取数据，也可以将数据写入字符串，兼具 istringstream 和 ostringstream 的功能。

#### stringstream

> stringstream 类兼具 istringstream 和 ostringstream 的功能，既可以从字符串中读取数据，也可以将数据写入字符串。(需要包含\<sstream\>ss头文件)

```c++
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
#define endl '\n'

int main() {
    string lines = "a 25 123456";
    stringstream inps(lines); // inps 是变量名
    int age;
    string name, id;
    inps >> name >> age >> id; // 将字符串中的数据存入定义好的变量中
    cout << "name:" << name << " age:" << age << " id:" << id << endl;
    
    stringstream outs; outs.str().reserve(1024); // 初始化并预留1024字节的空间
    int year = 2022, month = 3, day = 24;
    outs.width(4); outs.fill('0'); outs << year; // 设置下次输入的宽度为4，并用‘0’预先存储，向其中存入年份
    outs.width(2); outs.fill('0'); outs << month;
    outs.width(2); outs.fill('0'); outs << day;
    cout << outs.str() << endl;
    return 0;
}
```

---

## 正则表达式-regex

需要包括头文件\<regex>

### std::regex

表示一个正则表达式对象。正则表达式对象可以用来存储和表示一个特定的正则表达式模式。

- **.** ：匹配任意单个字符（换行符除外）。
- **$\left[\;\right]$** ：匹配方括号内的任意一个字符。例如，[abc] 匹配 'a'、'b' 或 'c'。
- **^** ：在方括号内使用时，表示取反。例如，[^abc] 匹配除 'a'、'b'、'c' 之外的任意字符。
- **\d** ：匹配任意数字，等价于 [0-9]。
- **\s** ：匹配任意空白字符，包括空格、制表符、换行符等。
- **\w** ：匹配任意字母、数字或下划线，等价于 [a-zA-Z0-9_]。

**常用量词** 

- **\*** : 匹配前面的模式零次或无数次。
- **\+** : 匹配前面的模式一次或多次。
- **？** ： 匹配前面的模式零次或一次。
- **{n}** : 匹配前面的模式恰好 n 次。
- **{n,}** : 匹配前面的模式 至少 n 次。
- **{n, m}** : 匹配前面的模式至少 n 次至多 m 次。

**常用锚点** 

- **^** : 匹配字符串的开始。
    - ^abc 可以匹配以"abc"开头的字符串。
- **$** : 匹配字符串的结束。
    - abc$ 可以匹配以"abc"结尾的字符串。
- **\b** : 匹配单词的边界。
    - \bcan\b 可以匹配单独的"can"单词。
- **\B** : 匹配非单词边界。
    - \Bcan\B 可以匹配于其他单词内部的"can"。

**分组**
分组用来将模式的匹配结果进行分组，并对每个分组进行单独的处理。用 **( )** 表示。

- (ab)+ 可以匹配 "ab" "abab" "ababab"。
- (a|b) 可以匹配 "a" 或者 "b"。 

```c++
#include <iostream>
#include <regex>
using namespace std;

int main(void){
    string str = "This is a string. 123456789";
    // 创建一个正则表达式对象，pattern为变量名。
    regex pattern("\\d+"); // 匹配一个或多个数字， 括号中也可写作 "[0-9]+"

    return 0;
}
```

### std::regex_match

用于检查整个字符串是否于表达式匹配。

```c++
// 形式 1：仅检查是否匹配
bool regex_match(const char* str, const std::regex& re);
bool regex_match(const std::string& str, const std::regex& re);

// 形式 2：检查匹配并存储匹配结果
template <class BidirIt, class Alloc, class CharT, class Traits>
bool regex_match(
                BidirIt first, 
                BidirIt last,
                std::match_results<BidirIt, Alloc>& m,
                const std::basic_regex<CharT, Traits>& e,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );
```

例子：

```c++
#include <bits/stdc++.h>
using namespace std;

int main(void) {
    std::string str = "abc123";
    std::regex pattern("abc\\d+");

    if (std::regex_match(str, pattern)) {
        std::cout << "整个字符串匹配正则表达式" << std::endl;
    } else {
        std::cout << "整个字符串不匹配正则表达式" << std::endl;
    }

    return 0;
}
```

### std::regex_search

用于在字符串中查找第一个与正则表达式匹配的子串。

```c++
// 形式 1：仅检查是否存在匹配的子串
bool regex_search(const char* str, const std::regex& re);
bool regex_search(const std::string& str, const std::regex& re);

// 形式 2：查找匹配并存储匹配结果
template <class BidirIt, class Alloc, class CharT, class Traits>
bool regex_search(
                BidirIt first, 
                BidirIt last,
                std::match_results<BidirIt, Alloc>& m,
                const std::basic_regex<CharT, Traits>& e,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );
```

例子：

```c++
#include <bist/stdc++.h>
using namespace std;

// 普通版
void solve_normal(void){
    string str = "This is a string. 1234567890";
    regex pattern("[0-9]+"); // 匹配一个或多个数字
    smatch matches; // 用于存储匹配的结果。
    if(regex_search(str, matches, pattern)){
        cout << "Found number: " << matches.str() << endl;
    }else{
        cout << "No match found." << endl;
    }
}
// 加强版
void solve_difficult(void){
    string str = "This is a string. 123-456-7890";
    regex pattern("(\\d{3})-(\\d{3})-{\\d{4}}");
    smatch matches;
    if(regex_search(str, matches, pattern)){
        // matches.str(0) 表示返回整个匹配的字符串。
        cout << "Found number: " << matches.str(0) << endl; 
        // matches.str(i) (i > 0) 表示返回第 i 个捕获组的匹配结果。
        for(size_t i = 1; i < matches.size(); i ++){
            cout << "捕获组" << i << ": " << matches.str(i) << endl;
        }   
    }else{
        cout << "Not find." << endl;
    }
}

int main(void){
    cout << "普通版：" << endl;
    solve_normal();

    cout << "加强版: " << endl;
    solve_difficult();

    return 0;
}
```

注：**std::smatch** 对象可以存储多个匹配结果，包括整个匹配的字符串以及各个捕获组的匹配结果。
**smatch 的常用成员函数：**

- **smatch.size()** : 返回匹配结果的数量，包括整个匹配和各个匹配组。
- **smatch.empty()** : 判断匹配结果是否为空。true 表示为空；
- **smatch.prefix()** : 返回匹配结果之前的字符串。
- **smatch.suffix()** : 返回匹配结果之后的字符串。
- **smatch.position()** : 返回匹配结果在原字符串的起始位置。
    - **smatch.position(i)** : (i>0)返回第 i 个捕获组匹配结果在原数组中的起始位置。  
- **smatch.length()** : 返回匹配结果的字符长度。

注：若可没有匹配结果，调用 position 会报错。预先判断是否有匹配结果。

### std::regex_replace

用于替换字符串中与正则表达式匹配的子串。

```c++
// 形式 1：返回替换后的字符串
template <class OutputIt, class BidirIt, class CharT, class Traits, class ST, class SA>
OutputIt regex_replace(
                OutputIt out,
                BidirIt first,
                BidirIt last,
                const std::basic_regex<CharT, Traits>& e,
                const std::basic_string<CharT, ST, SA>& fmt,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );

// 形式 2：返回替换后的字符串
template <class CharT, class Traits, class ST, class SA, class Fmt>
std::basic_string<CharT, ST, SA> regex_replace(
                const std::basic_string<CharT, ST, SA>& s,
                const std::basic_regex<CharT, Traits>& e,
                Fmt&& fmt,
                std::regex_constants::match_flag_type flags = std::regex_constants::match_default
                );

```


```c++
#include <bits/stdc++.h>;

int main() {
    std::string str = "hello, 123 world";
    std::regex pattern("\\d+");

    std::string result = std::regex_replace(str, pattern, "###");
    std::cout << "替换后的字符串: " << result << std::endl;

    return 0;
}
```

### sregex_iterator

主要用于遍历字符串中所有与给定正则表达式匹配的子串。

```c++
std::sregex_iterator(
                const BidirectionalIterator first, 
                const BidirectionalIterator last,
                const std::basic_regex<CharT, Traits>& re,
                std::regex_constants::match_flag_type flags =  std::regex_constants::match_default
                );
/*
first,last: 表示要搜索的字符串范围，通常是字符串的起始和结束迭代器。
re: 要匹配的正则表达式。
flags: 匹配标志，用于指定匹配的行为，默认为 std::regex_constants::match_default 。
*/
```

例子(不推荐，可能出错，建议用regex_search)：

```c++
#include <iostream>
#include <regex>
#include <string>

int main() {
    std::string text = "The cat sat on the mat. The cat is cute.";
    std::regex pattern("\\b(cat)\\b"); // 匹配整个单词 "cat"

    std::sregex_iterator it(text.begin(), text.end(), pattern);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        std::cout << "匹配到的子串: " << match.str() << "，起始位置: " << match.position() << std::endl;
        ++it;
    }

    return 0;
}

```



### 应用

#### 替换指定数量个匹配子串

```c++
#include <iostream>
#include <regex>
#include <string>

// 替换原字符串中指定数量的匹配结果
std::string replaceSpecificMatches(const std::string& input, const std::regex& pattern, 
                                   const std::string& replacement, int count) {
    std::string result = input;
    int replacedCount = 0;
    std::sregex_iterator it(result.begin(), result.end(), pattern);
    std::sregex_iterator end;

    while (it != end && replacedCount < count) {
        std::smatch match = *it;
        result.replace(match.position(), match.length(), replacement);
        // 重新创建迭代器，因为字符串已被修改
        it = std::sregex_iterator(result.begin(), result.end(), pattern);
        replacedCount++;
    }
    return result;
}

int main() {
    std::string text = "The cat sat on the mat. The cat is cute.";
    std::regex pattern("\\b(cat)\\b");
    std::string replacement = "dog";
    int replaceCount = 1;

    std::string newText = replaceSpecificMatches(text, pattern, replacement, replaceCount);
    std::cout << "替换 " << replaceCount << " 次后的字符串: " << newText << std::endl;

    return 0;
}    
```

## 前缀函数

定义 **$pi$** 为字符串 **$s$** 的前缀函数。
**$pi$**: $if$ `s[0~i]` 子串有相等的真前缀与真后缀，$then$ `pi[i]` 等于其中最长的一组的长度。
注：“真”代表非空。
例如：$s = abceabcf$
$pi\left[i\right]$ = {0, 0, 0, 0, 1, 2, 3, 0}

### template

朴素：$O(n^3)$

```c++
const int N = 1e5;
int pi[N] = {0}; // s 索引从0开始
for(int i = 1; i < len; i ++){  // 枚举子串结束位置位置，0跳过
    for(int j = i; j > 0; j --){ // 枚举真前后缀区间长度
        if(s.substr(0, j) == s.substr(i - j + 1, j)){
            pi[i] = j;
            break;
        }
    }
}
```

优化一：$O(n^2)$

相邻的前缀函数值最多增加1

```c++
const int N = 1e5;
int pi[N] = {0};
for(int i = 1; i < len; i ++){  // 枚举子串结束位置位置，0跳过
    for(int j = pi[i - 1] + 1; j > 0; j --){
        if(s.substr(0, j) == s.substr(i - j + 1, j)){
            pi[i] = j;
            break;
        }
    }
}
```

优化二：$O(n)$

`s[i + 1] == s[pi[i]]`

```c++
const int N = 1e5;
int pi[N] = {0};
for(int i = 1; i < len; i ++){
    int j = pi[i - 1];
    while(j > 0 && s[i] != s[j]){
        j = pi[j - 1];
    }
    if(s[i] == s[j]){
        j ++;
        pi[i] = j;
    }
}
```


## 字符串匹配-KMP

> KMP算法是一种在任何情况下都能达到 **$O(n + m)$** 复杂度的算法。

### template

```c++
const int N = 1e5 + 10;
int nxt[N];
void getNext(string s){
    // memset(nxt, -1, sizeof nxt); // 不必要
    int m = s.size();
    nxt[0] = -1;
    nxt[1] = 0;
    // i 表示当前要求 nxt 数组值的位置
    // cur 表示当前要和前一个字符比对的下标
    int i = 2, cur = 0;
    while(i < m){
        if(s[i - 1] == s[cur]){
            nxt[i ++] = ++ cur;
        }else if(cur > 0){
            cur = nxt[cur];
        }else{
            nxt[i ++] = 0;
        }
    }
}

int kmp(string s1, string s2){
    /*
    在 s1 中匹配 s2
    */
    int n = s1.size(), m = s2.size();
    int x = 0, y = 0;   // x 为 x1 中当前对比的位置，y 为 x2 中当前对比的位置
    while(x < n && y < m){
        if(s1[x] == s2[y]){
            x ++, y ++;
        }else if(y == 0){
            x ++;
        }else{
            y = nxt[y];
        }
    }
    return y == m ? x - y : -1;
}
```

---

## 字典树（前缀树）

### template(指针)

```c++
struct TrieNode{    // 字典树的节点
    vector<TrieNode*> nxt;  // 子节点
    bool isWord;    // 是否是字符串结尾
    TrieNode():nxt(26, nullptr), isWord(0){}    // 无参构造函数
};

struct TrieTree{    // 字典树
    TrieNode* root; // 根节点
    TrieTree():root(new TrieNode()){}   // 无参构造函数

    void insert(string s){  // 向字典树种添加字符串
        TrieNode* cur = root;   // 初始化光标
        for(char c : s){
            int inx = c - 'a';  // 获取将字母转换为 [0-25] 的数字
            if(!cur->nxt[inx]){ // 如果为空，则向后追加
                cur->nxt[inx] = new TrieNode();
            }
            cur = cur->nxt[inx];    // 光标向后移动
        }
        cur->isWord = true; // 该节点为字符串结尾
    }
    
    bool search(string s){ // 查找是否存在完成字符串 s，要求完整字符串，非前缀
        TrieNode* cur = root;
        for(char c : s){
            int inx = c - 'a';
            if(!cur->nxt[inx]) return false;
            cur = cur->nxt[inx];
        }
        return cur->isWord;
    }
    bool exist(string prefix){  // 查找是否存在前缀字符串 prefix
        TrieNode* cur = root;
        for(char c : prefix){
            int inx = c - 'a';
            if(!cur->nxt[inx]) return false;
            cur = cur->nxt[inx];
        }
        return true;
    }
};
```

### template(数组)

```c++
const int N = 1e5 + 10;

struct TrieTree{
    int tree[N][26], id, isWord[N];
    // tree:字典树    id:通过该编号来记录节点关系    isWord:是否是单词结尾 

    void insert(string s){
        int cur = 0;
        for(char c : s){
            int inx = c - 'a';
            if(!tree[cur][inx]){
                tree[cur][inx] = ++id;
            }
            cur = tree[cur][inx];
        }
        isWord[cur] = 1;
    }

    bool search(string s){
        int cur = 0;
        for(char c : s){
            int inx = c - 'a';
            if(!tree[cur][inx]) return false;
            cur = tree[cur][inx];
        }
        return isWord[cur];
    }
};
```

---

## 字符串最小表示

```c++
int minRepresentation(string s){
    int n = s.size();
    s += s; // 将字符串拼接成两倍长，方便处理循环位移
    int i = 0, j = 1, k = 0; // i 和 j 是两个起始位置，k 是当前比较的字符数
    while (i < n && j < n && k < n) {
        if (s[i + k] == s[j + k]) {
            k++; // 当前字符相等，继续比较下一个字符
            continue;
        }
        if (s[i + k] > s[j + k]) {
            i = i + k + 1; // i 的字典序更大，跳过当前比较
        } else {                
            j = j + k + 1; // j 的字典序更大，跳过当前比较
        }
        if (i == j) i ++; // 避免 i 和 j 重合
        k = 0; // 重置比较的字符数
        
    }
    return min(i, j); // 返回字典序最小的起始位置
}
```

# 数据结构

## 线段树

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



## 树状数组

> **求区间和。** 可以在 $O(nlogn)$ 的时间构建树状数组， $O(logn)$ 的时间更新指定节点数据和查询区间和。

```c++
template <typename T>
struct BIT{
    // 数组大小
    int n; 
    vector<T> tree;
    inline int lowbit(int x){return x & -x;}

    BIT(int sz) : n(sz), tree(sz + 1, T()){}

    template <typename U>
    void build(const vector<U>& arr){
        for(int i = 0; i < n; i ++){
	            update(i + 1, static_cast<T>(arr[i]));
        }
    }

    void update(int idx, T val){
        for(; idx <= n; idx += lowbit(idx)){
            tree[idx] += val;
        }
    }

    T query(int idx){
        T res = T();
        for(; idx > 0; idx -= lowbit(idx)){
            res += tree[idx];
        }
        return res;
    }
    
    T range_query(int lf, int rt){
        if(lf > rt || lf < 1 || rt > n) return 0;
        return query(rt) - query(lf - 1);
    }
};
```

## 单调栈

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

## 单调队列

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

