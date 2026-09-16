# 核心交换机配置

## 开启核心交换机的路由功能

1. 在配置中点击任意接口，使CLI 进入 `Switch(config-if)` 界面
2. 输入 `end` 命令，进入 `Switch#`
3. 输入 `conf` 进入 `Switch(config)#`
4. 输入 `ip routing` 命令，开启核心交换机的路由功能，后进行以下配置

## 配置 vlan 地址

1. 在配置中点击任意接口，使CLI 进入 `Switch(config-if)` 界面
2. 通过 `int vlan20` 命令进入 vlan20 配置
3. `ip address` 命令配置具体 vlan 的地址
   ```
   ip address ip地址 子网掩码
   ```
4. 输入 `no shutdown` 开启接口
5. 重复以上步骤

## 配置 物理接口 地址

1. 在配置中点击任意结构，使CLI 进入 `Switch(config-if)` 界面
2. 通过 `int f0/3` 命令进入指定物理接口配置
3. 由于 vlan 位于第二层(数据链路层)，配置物理接口需要进入第三层(网络层)进行配置，输入 `no siwtchport` 切换到第三层
4. 使用 `ip address` 命令配置指定接口的 ip 地址和子网掩码
5. 输入 `no shutdown` 开启接口

## 给物理各物理接口封装 交换协议 并 切换至 trunk 模式

1. 进入 `switch(config)#` 或 `switch(config-if)#` 后 输入 `int f0/1` 等进入配置
2. 输入 `switchport trunk encapsulation dotlq` 安装交换协议
3. 输入 `switchport mode trunk` 将接口切换至 `trunk` 模式

## 配置 enable 密码

1. 进入 `switch(config)#` 
2. 输入 `enable password 密码` 

# 路由器配置

## RIP 配置

在配置中选择 RIP 添加和其直接相连的各网段。

可以通过在 命令行中进入 特权模式 `Router#` 下输入 `show ip route` 命令查看网段。