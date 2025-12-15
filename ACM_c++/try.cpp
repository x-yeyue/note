# include <bits/stdc++.h>
using namespace std;
//#define int long long
#define ll long long
#define endl '\n'
#define pii pair<int, int>
#define show(arr){for(int i = 0; i < arr.size(); i ++){cout << arr[i] << " ";} cout << endl;}

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

void solve(void){
    int n; cin >> n;
    for(int i = 1; i <= n; i ++){
        cin >> arr[i];
    }
    merge_sort(1, n);
    for(int i = 1; i <= n; i ++){
        cout << arr[i] << " ";
    }cout << endl;
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
