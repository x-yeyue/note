## Scriptlet 脚本小程序

**JSP** 一共有三种 Scriptlet 脚本小程序:
1. Java 脚本段。可以写 Java 代码,定义局部变量、编写语句等
   生成的代码在 servlet 中 `service` 方法体中。
   ```jsp
   <%
       // 可以写 Java 代码
   %>
   ```
2. 声明。声明全局变量、方法、类等
   生成的代码在 servlet 的类体中。
   ```jsp
   <%!
       // 声明全局变量
   %>
   ```
3. 输出表达式。可以输出变量或自变量
   生成的代码在 servlet 中的 `service` 方法体中，相当于 `out.print()` 输出
   ```jsp
   <%= 数值 %>
   ```

演示:

```jsp
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<!DOCTYPE html>
<html>
<head>
  <title>JSP - Hello World</title>
</head>
<body>
  <%
    // 定义局部变量
    String str = "Hello Jsp";
    // 输出内容到控制台
    System.out.println(str);
    // 输出内容到浏览器
    out.print(str);
    out.print("------");
    // 输出全局变量
    out.write("全局变量:" + num);
  %>

  <%!
    int num = 10;
  %>

  <%=
    str
  %>
</body>
</html>
```

## include 静态包含

**格式**:
```jsp
<%@include file="url"%>
```

**特点**:
1. 将页面进行了直接的替换
2. 静态包含只会生成一个源码文件，最终的内容全在 `_jspService` 方法体中(源码文件中)
3. 不能出现同名变量(因为会整合到同一个文件内)
4. 运行效率高一点点。耦合较高。

## include 动态包含

**格式**:
```jsp
<jsp:include page="url"></jsp:include>
```

**特点**:
1. 动态包含相当于方法的调用
2. 动态包含会生成多个源码文件
3. 可以定义同名变量
4. 效率高，耦合度低

**注**: 当动态包含不需要传递参数时， `include` 双标签之间不要有任何内容，包括换行和空格。

**传递参数**:
```jsp
<jsp:include page="url">
	<jsp:param name="参数名" value="参数值"/>
</jsp:include>
```

**注**: `name` 属性不支持表达式，`value` 属性支持表达式。

## JSP 的四大域对象

> 在 JSP 中提供了四种属性的保存范围，所谓的属性保存范围，指的是一个设置的对象，可以再多少个页面中保存并可以继续使用。

### page 范围

`pageContext`: 只在一个页面中保存属性，跳转(`<jsp: forward>`)之后无效。

### request 范围

`request`: 只在一次请求中保存，服务器跳转后依然有效。

服务器跳转(`<jsp: forward>`)有效，客户端跳转(*超链接*)无效。

### session 范围

`session`: 在一次会话范围中，无论何种跳转都可以使用。

无论客户端还是服务端都可以，但重新开启一个新的浏览器，则无法取得之前设置的 `session`。

对于服务器，每一个连接到他的客户端都是一个 `session`。

### application 范围

`application`: 在整个服务器上保存，每一个用户(`session`)都可以直接访问。

服务器重启，则所有属性消失。


### 常用命令

#### 增删
```java
// 设置属性的名称及内容
public void setAttribute(String name, Object o)

// 根据属性名称取数据
public Object getAttribute(String name)

// 删除指定的属性
public void removeAttribute(String name)
```

**示例**:

```jsp
<%
	// 设置 page 范围的域对象
	pageContext.setAttribute("name1", "张三");
	
	// 设置 request 范围的域对象
	request.setAttribute("name2", "李四");
	
	// 设置 session 范围的域对象
	session.setAttribute("name3", "王五");
	
	// 设置 application 范围的域对象
	application.setAttribute("name4", "赵六");
%>
```

#### 跳转

**服务端跳转**:
```jsp
<jsp:forward page="url"></jsp:forward>
```

**客户端**: 超链接

## EL 表达式

**作用**: 简化 JSP 代码。

**格式**: `${域对象名称}`

**操作对象**: 域对象，不能操作局部变量。

**注**:
1. 如若 EL 表达式获取域对象的值为空，默认显示空字符串。
2. EL 表达式默认从小到大范围寻找，找到就返回，否则显示空字符串。
3. 查找指定域对象的值
   `pageScope`,`requestScope`,`sessionScope`,`applicaionScope`。
   `${pageScope.username}` 依此类推。

### 获取数据

**获取List**
```jsp
// 获取 list 的 size
${list.size()}

// 获取 list 指定下标的值
${list[index]}
```

**获取map**
```jsp
// 获取 map 中指定 key 的 value
${map.key}
${map["key"]}
```

**获取 javaBean**
```jsp
${user.id}
${User.getId()} // 要提供对应的 get 方法
```

### empty

判断域对象是否为空: `${empty name}`
- 空，返回 `true`
- 非空，返回 `false`

是否非空: `${!empty name}`

如果域对象是 **字符串**:
- 不存在的域对象: `true`
- 空字符串: `true`
- `null`: `true`
- 有值: `false`

```jsp
<%  
    request.setAttribute("str1", "aaa");  
    request.setAttribute("str2", "");  
    request.setAttribute("str3", null);  
%>  
  
${empty str}  // true
${empty str1}  // false
${empty str2}  // true
${empty str3}  // true
```

如果域对象是 `List`:
- `null`: `true`
- 没有长度(`size == 0`): `true`

如果域对象是 `Map`:
- `null`: true
- 空 `map` 对象: `true`

如果域对象是 `JavaBean`:
- `null`: `true`
- 空对象: `false`

### 相等

```jsp
${a == b}
${a eq b}
${a == 5}
${c == "aa"}
```

### 算数运算

```jsp
${a + b}
${a / b}
${a div b}
```

### 大小比较

```jsp
${a > b}
${a > b && b > 5}
%{a > b || b > 5}
```

## JSTL

### 引入依赖(Maven)
```xml
<!-- JSTL API 接口定义 -->
<dependency>  
    <groupId>jakarta.servlet.jsp.jstl</groupId>  
    <artifactId>jakarta.servlet.jsp.jstl-api</artifactId>  
    <version>3.0.2</version>  
</dependency>  
<!-- JTSTL 实现(GlassFish 提供) -->
<dependency>  
    <groupId>org.glassfish.web</groupId>  
    <artifactId>jakarta.servlet.jsp.jstl</artifactId>  
    <version>3.0.1</version>  
</dependency>
```
### 引入核心标签库

```jsp
// prefix="c" 表示后续使用 <c:xxx> 前缀调用核心标签
<%@taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
```

引入后可使用 `<c:if>`、`<c:forEach>`、`<c:choose>` 等核心标签。

### 条件动作标签

> `if`,`choose`, `when`,`otherwise`。

#### if

`if` 标签先对某个条件进行测试，如果条件运算结果为 `true`，则处理它的主体内容，测试结果保存在一个 `Boolean` 对象中，并创建一个 *限域变量* 来引用 `Boolean` 对象。可以用 `var` 属性设置 限域变量名，利用 `scope` 属性来指定其作用范围。

**语法**:

```jsp
<c:if test="<boolean>" var="<string>" scope="<string>">
	...
</c:if>

// 没有 else，需要则设置相反条件
```

**属性**:

- `test`: 条件。必要。
- `var`: 用于存储条件结果的变量(限域变量名)。
- `scope`: `var` 属性的作用域(`page 默认 | request | session | application`)。

#### choose、when、otherwise

**语法**:
```jsp
<c:choose>
	<c:when test="<boolean>">
		...
	</c:when>
	<c:when test="<boolean>">
		...
	</c:when>
	...
	...
	...
	<c:otherwise>
		...
	</c:otherwise>
</c:choose>
```

**属性**:
- `choose` 没有属性。
- `when` 只有一个 `test` 属性，且必要。
- `otherwise` 没有属性。

**注意**:
- `choose` 标签和 `otherwise` 标签没有属性，而 `when` 标签必须有一个 `test` 属性。
- `choose` 标签中必须包含至少一个 `when` 标签，可以没有 `otherwise` 标签。
- `otherwise` 标签必须设置在最后一个 `when` 标签之后。
- `choose` 标签中之只能设置 `when` 标签域 `otherwise` 标签。
- `when` 标签和 `otherwise` 标签中可以嵌套其他标签。
- `otherwise` 标签会在所有的 `when` 标签都不执行时才会执行。

#### forEach

**语法**:
```jsp
<c:forEach
	items="<object>"
	begin="<int>"
	end="<int>"
	step="<int>"
	var="<string>"
	varStatus="<string>" >
</c:forEach>
```

**属性**：
- `items`: 要被循环的数据。
- `begin`: 开始的元素(0 开始)。
- `end`: 最后一个元素。
- `step`: 步长。
- `var`: 当前条目的变量名。
- `varStatus`: 循环状态的变量名。
  - *index*: 从 0 开始迭代索引。
  - *count*: 从 1 开始迭代计数。
  - *first*: 当前这轮迭代是否为第一次。
  - *last*: 当前这轮迭代是否为最后一次。

**示例**:

```jsp
<c:forEach begin="开始数" end="结束数" step="步长" var="限域变量名">

</c:forEach>

->

<c:forEach begin="0" end="10" var="i">
	标题${i}<br>
</c:forEach>
```

```jsp
<c:forEach items="被循环的集合" var="限域变量名">
	
</c:forEach>

->

<%
	List<String> list = new ArrayList<>();
	for (int i = 1; i <= 10; i ++){
		list.add("A:" + i);
	}
	pageContext.setAttribute("li", list);
%>

<c:forEach items="${li}" var="i">
	${i} &nbsp;
</c:forEach>
```

#### formatNumber

用于格式化数字、百分比、货币。

**语法**:
```jsp
<fmt:formatNumber
	value="<string>"
	type="<string>"
	var="<string>"
	scope="<string>" />
```

**属性**:
- `value`: 要显示的数值(必要)。
- `type`: NUMBER, CURRENCY, PERCENT。默认 NUMBER。
- `var`: 存储格式化数字的变量(*page*|*request*|*session*|*application*)。默认 Print to page。
- `scope`: var 属性的作用域。默认 page。

**注**: 如果设置了 `var` 属性，则格式化后的结果不会输出，需要通过 *EL* 表达式获取 `var` 对应的限域变量名。

**示例**:
```jsp
<fmt:formatNumber val="10" type="number" var="num" /> ${num} <br>
<fmt:formatNumber val="10" type="precent" /> <br>
<fmt:formatNumber val="10" type="currency" /> <br>
<!-- 设置时区 -->
<fmt:setLocale value="en_US" />
<fmt:formatNumber val="10" type="currency" /> <br>
```

#### formatDate

格式化 `Date` 型日期。

**语法**:
```jsp
<fmt:formatDate
	value="<string>"
	type="<string>"
	dateStyle="<string>"
	timeStyle="<string>"
	pattern="<string>"
	timeZone="<string>"
	var="<string>"
	scope="<string>" />
```

**属性**:
- `val`: 要显示的日期(必要)。
- `type`: *DATE*, *TIME*, *BOTH*。默认 *date*
  - `date`: 日期型。年月日
  - `time`: 时间型。时分秒
  - `both`: 日期时间型。
- `dateStyle`: *FULL*, *LONG*, *MEDIUM*, *SHORT*, *DEFAULT*。默认 *default*
- `timeStyle`: *FULL*, *LONG*, *MEDIUM*, *SHORT*, *DEFAULT*。默认 *default*
- `pattern`: 自定义格式模式。**y** **M** **d** **h**(12小时制) **H**(24小时) **m** **s**
- `timeZone`: 显示日期的时区。