11001
11100 -> 11101

11101
11110 -> 111111

100  010  001
010  001  100
110  011  101

1111000110010
0111100011001

邻接矩阵
邻接表
链式前向星

  a b c 
a 0 1 0
b 0 0 3
c 0 3 0
```c++
[
[]
[]
[]
]
```

a = [b]
b = [a,dsfafdgdgdfds ]
c = []
```c++
[dfgff   ] 
[     d f  fds ffg    ]
n * m
vector<vector<int> > mat(100000, vector<int>{})

```

```c++
next_permutation()

vector<bool> vis(n + 1, false);
vector<int> permutation;
void get_permutation(int idx){
	if(idx = n){
		
		return;
	}
	for(int i = 1; i <= n; i ++){
		if(!vis[i]){
			vis[i] = true;
			permutation.emplace_back(i);
			get_permutation(idx + 1);
			permutation.pop_back();
			vis[i] = false;
		}
	}
}
123
132
213
231
312
321

vector<int> arr(3);
do{
	
}while(next_permutaion(arr))
```