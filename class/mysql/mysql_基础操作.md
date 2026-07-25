# $MYSQL$

## 启动与关闭 $MYSQL$

### 启动 $MySQL$

`sudo mysql -u root -p`

- **sudo**: 管理员权限。
- **mysql**: 调用 $MySQL$ 终端。
- **-u root**: 以 $root$ 的身份登录。
- **-p**: 输入密码。

### 关闭 $MySQL$

#### 关闭 $MySQL$ 服务

##### 在 $MySQL$ 命令行

`SHUTDOWN` 命令(需要为 $root$ 用户)

##### 通用

`sudo systemctl stop mysql`

- **sudo**: 管理员权限。
- **systemctl stop mysql**: 让系统停止 **MySQL** 服务。

#### 退出 $MySQL$ 界面

`exit;`

## 数据库基本操作

### 创建数据库


#### 命名规则

1. 一般由字母和下划线组成，不允许有空格，可以是英文单词、英文短语或相应缩写；
2. 不允许是 $MYSQL$ 关键字；
3. 长度最好不超过 $128$ 位；
4. 不能与其他数据库同名。



## 存储引擎、数据类型和字符集

### 存储引擎

**定义：** 存储引擎是 $MYSQL$ 体系结构的重要组成部分。
**作用：** 指定表的类型，规定表如何存储和索引数据、是否支持事务等。

#### 查看 $MYSQL$ 支持的存储引擎

`SHOW ENGINES \G`

- **\G** ：讲查询结果按列显示。

![SHOW_ENGINES_\G](img/mysql/SHOW_ENGINES_G.png "SHOW_ENGINES_\G.png")

常用参数：

- **Engine:** 表示存储引擎的名称；
- **Support:** 表示 $MYSQL$ 是否支持此存储引擎；
- **Comment:** 表示关于此存储引擎的评论；
- **Transactions:** 表示此存储引擎是否支持事务；

#### 查看 $MYSQL$ 默认存储引擎

`SHOW VARIABLES LIKE 'default_storage_engine';`

#### 常用的存储引擎

1. **InnoDB**
$InnoDB$ 为 $MYSQL$ 提供具有 *提交*、*回滚*、*崩溃恢复*能力 和 *多版本并发控制* 的事务安全型表，能够高效地处理大量数据。
适用于需要 ***事务支持*、*高并发*、*数据更新频繁*、*对数据的一致性* 和 *完整性* 要求较高**的 *计费系统* 和 *财务系统* 等。

2. **MyISAM**
$MyISAM$ 存储引擎基于 $ISAM$，并对其进行扩展。
具有较高的 *插入* 和 *查询* 速度，但**不支持 *事务* 和 *外键***。
每个使用 $MyISAM$ 存储引擎创建的数据表都会生成 $3$ 个文件，文件名和数据表名称相同，但扩展名不同，各个文件及其作用分别如下：
   3. **$.frm$:** 存储表定义。
   4. **$.MYD$:** 存储数据。
   5. **$.MYI$:** 存储索引。
对 ***事务完整性没有要求*、*并发相对较低*、*数据更新不频繁*、*以读为主* 和 *对数据一致性要求不高*** 的数据表，推荐使用 $MyISAM$ 存储引擎。

6. **MEMORY**
$MEMORY$ 存储引擎主要用于 ***内容变化不频繁*** 的表。
由于 $MEMORY$ 存储引擎是**将数据存储到 *内存*** 中，所以 *太大的表无法使用此存储引擎*。
对于 ***数据更新不频繁*、*存货周期不长* 和 *需要对数据统计结果进行z分析*** 的数据表可以使用 $MEMORY$ 存储引擎。

### 数据类型

#### 数值类型

> **整数类型**
>
> - 整数是**有符号数**，在有符号的情况下，其所占字节的第一位为符号位(**$0$ 代表整数，$1$ 代表负数**)。
> - 如果插入的数据 **超出所选数据类型的存储范围**，或者**为其他类型的数据**时，系统会提示 “**Out \; of \; range$**” 的错误信息。
> - 在定义整数类型字段时，在类型名称后面的括号重加入数字，可以指定宽度。
> - $n :=$ 整形类型字段宽度，$x :=$ 输入的整数宽度。**$if \; x \lt n \;\; then \;\; showWidth = n$** 缺少部分用空格填充。**$if \; x \gt n \; \land \; x \lt maxLenOfIntRange \;\; showLen = x$**；  

|   数据类型    |  所占字节  |                 存储类型(有符号)                  |        存储范围(无符号)         |
| :-------: | :----: | :----------------------------------------: | :----------------------: |
|  TINYINT  | $1$ 字节 |                 -128 ~ 127                 |         0 ~ 255          |
| SMALLINT  | $2$ 字节 |               -32768 ~ 32767               |        0 ~ 65535         |
| MEDIUMINT | $3$ 字节 |             -8388608 ~ 8388607             |       0 ~ 16777215       |
|    INT    | $4$ 字节 |          -2147483648 ~ 2147483647          |      0 ~ 4294967295      |
|  BIGINT   | $8$ 字节 | -9223372036854775808 ~ 9223372036854775807 | 0 ~ 18446744073709551615 |

> **UNSIGNED 关键字**：在类型后加 `UNSIGNED` 可取消符号位，使整数只能存储非负数，从而扩大正数的存储范围。例如 `INT UNSIGNED` 的范围为 $0 \sim 4294967295$。

> **显示宽度与 ZEROFILL**：类型后括号中的数字（如 `INT(5)`）仅影响显示宽度（配合 `ZEROFILL` 时用 `0` 填充），**不影响存储范围和实际存储大小**。`ZEROFILL` 会自动为该列添加 `UNSIGNED` 属性。

#### 浮点类型

> 浮点类型用于存储 **近似值**，存在精度损失，不适合存储精确数值（如金额）。

|  数据类型   |  所占字节  |        精度说明         |          适用场景           |
| :-----: | :----: | :-----------------: | :---------------------: |
|  float  | $4$ 字节 |  短小数（约 $7$ 位有效数字）   |       精度要求不高的科学计算       |
| double  | $8$ 字节 | 较长小数（约 $15$ 位有效数字）  |        精度要求较高的计算        |
| decimal |  可变字节  | 精确小数（最大 $65$ 位有效数字） | **金额、价格等需要精确计算的场景（推荐）** |

> **decimal 用法**：`decimal(M, D)` 其中 $M$ 为总位数（精度），$D$ 为小数位数（标度）。例如 `decimal(10, 2)` 表示最多 $10$ 位数字，其中 $2$ 位小数，即最大可存储 `99999999.99`。
> **float/double vs decimal**：`float` 和 `double` 采用二进制浮点运算，存在精度丢失问题（例如 `0.1 + 0.2 != 0.3`）。涉及精确计算时**务必使用 `decimal`**。

#### 字符串类型

> 字符串类型用于存储文本数据。选择合适类型的关键在于：是否定长、数据量大小、是否需要全文检索。

|    数据类型    |       存储方式        |        最大长度        |           说明           |     适用场景     |
| :--------: | :---------------: | :----------------: | :--------------------: | :----------: |
|    char    |     定长（不足补空格）     |      $255$ 字符      | 读取速度快，但浪费空间（始终占用声明的长度） |    固定长度文本    |
|  varchar   | 变长（额外 $1$ 字节记录长度） | $65535$ 字节（受行大小限制） |     节省空间，最常用的字符串类型     | 变长文本（姓名、地址等） |
|  tinytext  |        变长         |      $255$ 字符      |  与 tinyblob 类似，仅存储文本   |     短文本      |
|    text    |        变长         |     $65535$ 字符     |  不能设置默认值，不参与索引（除前缀索引）  |  文章、评论等长文本   |
| mediumtext |        变长         |   $16777215$ 字符    |         中等长度文本         |   较长的文本内容    |
|  longtext  |        变长         |  $4294967295$ 字符   |          超长文本          |  小说、日志等超大文本  |

> **char vs varchar**：
> - `char(10)` 存储 `'abc'`，实际占用 $10$ 个字符的空间（末尾补空格）；存取时会**自动去除尾部空格**。
> - `varchar(10)` 存储 `'abc'`，实际占用 $3 + 1 = 4$ 个字节（额外 $1$ 字节记录长度）。
> - **char** 适合长度几乎不变的字段（如 MD5 值固定为 $32$ 位、性别代码 `M`/`F`）。
> - **varchar** 适合长度变化较大的字段。

> **binary / varbinary**：分别对应 `char` / `varchar` 的二进制版本，用于存储二进制数据（如文件内容、加密密钥）。

#### 日期与时间类型

|   数据类型    | 所占字节 |          格式           |            是否自动更新             |     适用场景      |
| :-------: | :--: | :-------------------: | :---------------------------: | :-----------: |
|   year    | 1 字节 |        `YYYY`         |               否               |     仅存储年份     |
|   date    | 3 字节 |     `YYYY-MM-DD`      |               否               |  仅存储日期（生日等）   |
|   time    | 3 字节 |      `HH:MM:SS`       |               否               |  仅存储时间（课程时间）  |
| datetime  | 8 字节 | `YYYY-MM-DD HH:MM:SS` |               否               | 日期+时间（创建时间等）  |
| timestamp | 4 字节 | `YYYY-MM-DD HH:MM:SS` | **是**（默认 `CURRENT_TIMESTAMP`） | 需要自动记录更新时间的场景 |

> **datetime vs timestamp**：
> - `datetime` 占 $8$ 字节，范围为 `1000-01-01 00:00:00` ~ `9999-12-31 23:59:59`，**不受时区影响**，存什么读什么。
> - `timestamp` 占 $4$ 字节，范围为 `1970-01-01 00:00:01 UTC` ~ `2038-01-19 03:14:07 UTC`，**会随服务器时区转换**，存储时转为 UTC，读取时转为当前时区。
> - 推荐用 `datetime` 存业务时间（如订单创建时间），用 `timestamp` 做记录追踪（如 `updated_at`）。

> **自动初始化与更新**：`timestamp` 和 `datetime` 支持自动初始化和自动更新：
> ```mysql
> -- 创建时自动填充当前时间，更新时自动刷新
> create_time datetime default current_timestamp comment '创建时间',
> update_time datetime default current_timestamp on update current_timestamp comment '更新时间'
> ```

## 数据库操作

```mysql
-- 查询 所有数据库
show databases;

-- 查询 当前数据库
select database();

-- 使用/切换 数据库
user 数据库名;

-- 创建数据库
create database [if not exists] 数据库名 [default charset utf8mb4];

-- 删除数据库
drop database [if exists] 数据库名;
```

### 约束条件

#### 列级完整性约束

- `not null / null`: 不允许为空 / 允许为空。
- `unique`: 唯一性约束，不允许重复。
- `default`: 缺省值约束，将字段中使用频率最高的字段值设置，为该列的缺省值。
- `auto_increment`: 自增关键字。

#### 表级完整性约束

- `unique`: 字段组合不允许重复。
- `primary key`: 主键约束。一个可放在列级完整约束。
  `primary key(字段1 [, 字段2...])`
- `foreign key`: 外键约束。如有多个需要依次单独列出。
  `foreign key(外键字段名) references 对应主键所在表 (对应主键字段名)`


### 创建数据表

```mysql
create table [库名] 表名(
	属性名 属性类型 [列级完整约束条件] [comment 注释]
	[, ...]
	[, 表级完整性约束]
)[comment 表注释];
```



### 查看表

```mysql
-- 查询当前数据库的所有表
show tables;

-- 查询表结构
descibe 表名;
desc 表明;

-- 查询建表语句
show create table 表明;
```

### 修改表结构

```mysql
-- 添加字段
alter table 表名 add 字段名 类型 [comment 注释] [约束];

-- 修改字段类型
alter table 表名 modify 字段名 新数据类型;

-- 修改字段名和字段类型
alter table 表名 change 旧字段名 新字段名 类型 [comment 注释] [约束];

-- 删除字段
alter table 表名 drop column 字段名;

-- 修改表名
alter table 表名 rename to 新表名;

-- 添加主键
alter table 表名 add primary key (字段名 [, ...]);

-- 添加外键约束
alter table 表名 add foreign key (字段名) references 对应外键所在表(对应字段名) on 约束;

-- 删除外键约束
alter table 表名 drop foreign key 约束;
```

### 删除表

```mysql
drop table [if exists] 表名;
```

### 创建索引

在已经建好的表上创建索引。
`create index 索引名 on 表名 (属性名 [(长度)][asc|desc])`

### 创建视图

`create view name as select .....;`

## 命令

- `between num_a and num_b` 在 $[num\_a, num\_b]$ 之间。
- `distinct` 去重。
- `like "匹配模式"`: `%`任意个字符，`_`一个字符。

### 添加数据

```mysql
insert into 表名(字段名1, 字段名2, ...) values(值1, 值2,...);
insert into 表名 values(值1, 值2, ..., 值n);

insert into 表名(字段名1, 字段名2, ...) values
	(值1, 值2,...), 
	(值1, 值2,...),
	...;
insert into 表名 values
	(值1, 值2, ..., 值n),
	(值1, 值2, ..., 值n),
	...;
```

### 更新数据

`update 表名 set 字段名 = 表达式... [where ...];`

### 删除数据

`delete from 表名 [where ...];`

### 查询语句

```mysql
select 
	字段列表 
from 
	表名列表 
where 
	条件列表 
group by 
	分组字段列表 
having 
	分组后条件列表 
order by 
	排序字段列表 
limit 
	分页参数

-- 查询多个字段
select 字段1，字段2，字段3 from 表名;

-- 查询所有字段
select * from 表名;

-- 为查询字段设置别名，as 关键词可以省略
select 字段1 [as 别名1], 字段2 [as 别名2] from 表名;

-- 去除重复记录
select distinct 字段列表 from 表名;
```

#### 条件查询

```mysql
select 字段列表 from 表名 where 条件列表;
```

|       比较运算       |               功能               |
| :--------------: | :----------------------------: |
|        >         |               大于               |
|        >=        |              大于等于              |
|        <         |               小于               |
|        <=        |              小于等于              |
|        =         |               等于               |
|     <> 或 !=      |              不等于               |
| between...and... |         在某个范围之内(包含边界)          |
|     in(...)      |       在 in 之后的列表中的值，多选一        |
|     like 占位符     | 模糊匹配( `_` 匹配单个字符, `%` 匹配任意个字符) |
|     is null      |             是 null             |
|     and 或 &&     |               并且               |
|    or 或 \|\|     |               或者               |
|     not 或 !      |               非                |

#### 分组查询

*聚合函数*: 将一列数据作为一个整体，进行纵向计算。

|  函数   |  功能  |
| :---: | :--: |
| count | 统计数量 |
|  max  | 最大值  |
|  min  | 最小值  |
|  avg  | 平均值  |
|  sum  |  求和  |

**注**: null 不参与 聚合函数的统计。

```mysql
select 字段列表 from 表名 [where 条件列表] group by 字段分组名 [having 分组后过滤条件];
```

#### 排序查询

```mysql
select 字段列表 from 表名 [where 条件列表] [group by 字段分组名 [having 分组后过滤条件]] order by 排序字段 排序方式;

-- 排序方式: asc 默认 升序, desc 降序
```

#### 分页查询

```mysql
select 字段 from 表名 [where 条件] [group by 分组字段 having 过滤条件] [order by 排序字段] limit 起始索引, 查询记录数;
-- 起始索引从 0 开始
```
### 多表查询

- 等值连接
  `from 表1 join 表2 on 表1.字段 = 表2.字段`
- 自然连接
  `from 表1 natural join 表2`

#### 内连接

内连接查询的是两张表交集部分的内容。

```mysql
-- 隐式内连接
select 字段列表 from 表1, 表2 where 连接条件 ...;

-- 显示内连接
select 字段列表 from 表1 [inner] join 表2 on 连接条件 ...;
```

样例:

```mysql
-- A. 查询所有员工的ID, 姓名 , 及所属的部门名称 (隐式、显式内连接实现)
-- 隐式
select emp.id, emp.name, dept.name from emp, dept where dept.id = emp.dept_id;
-- 显示
select emp.id, emp.name, dept.name from emp join dept on dept.id = emp.dept_id;


-- B. 查询 性别为男, 且工资 高于8000 的员工的ID, 姓名, 及所属的部门名称 (隐式、显式内连接实现)
-- 隐式
select emp.id, emp.name, dept.name from emp, dept where emp.dept_id = dept.id && emp.gender = 1 && emp.salary > 8000;
-- 显示
select emp.id, emp.name, dept.name from emp join dept on emp.dept_id = dept.id where emp.gender = 1 && emp.salary > 8000;
```

#### 外连接

外连接分为 左外连接 和 右外连接。

左外连接包含所有左表数据和两表相交的内容。

右外连接同理。

```mysql
-- 左外连接
select 字段列表 from 表1 left [outer] join 表2 on 连接条件 ...;

-- 右外连接
select 字段列 from 表2 right [outer] join 表2 on 连接条件 ...;
```

样例:

```mysql
-- A. 查询员工表 所有 员工的姓名, 和对应的部门名称 (左外连接)
select emp.name, dept.name from emp left join dept on emp.dept_id = dept.id;

-- B. 查询部门表 所有 部门的名称, 和对应的员工名称 (右外连接)
select dept.name, emp.name from emp right join dept on emp.dept_id = dept.id;

-- C. 查询工资 高于8000 的 所有员工的姓名, 和对应的部门名称 (左外连接)
select emp.name, dept.name from emp left join dept on emp.dept_id = dept.id where emp.salary > 8000;
```

#### 子查询

SQL 语句中嵌套 select 语句，称为嵌套查询，又称为子查询。

```mysql
select * from 表1 where column1 = (select column1 from 表2 ...);
```

子查询外部的语句可以是 `insert`，`update`，`delete`，`select` 的任何一个。

分类:
- 标量子查询: 返回结果为单个值。
- 列子查询: 返回结果为一列。
- 行子查询: 返回结果为一行。
- 表子查询: 返回的结果为多行多列。

样例:

```mysql
-- 标量子查询
-- A. 查询 最早入职 的员工信息
select * from emp where emp.entry_date = (
	select min(entry_date) from emp
);

-- B. 查询在 "阮小五" 入职之后入职的员工信息
select * from emp where entry_date > (
	select entry_date from emp where name = "阮小五"
);


-- 列子查询
-- A. 查询 "教研部" 和 "咨询部" 的所有员工信息
select * from emp where dept_id in (
	select id from dept where name in ("教研部", "咨询部")
);


-- 行子查询
-- A. 查询与 "李忠" 的薪资 及 职位都相同的员工信息 ;
select * from emp where (salary, dept_id) = (
	select salary, dept_id from emp where name = '李忠'
);


-- 表子查询
-- A. 获取每个部门中薪资最高的员工信息
select * from emp join (
	select dept_id, max(salary) max_salary from emp group by dept_id
) temp on emp.dept_id = temp.dept_id && emp.salary = temp.max_salary;
```