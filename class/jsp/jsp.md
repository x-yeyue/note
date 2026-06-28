## Scriptlet 脚本小程序

**JSP** 一共有三种 Scriptlet 脚本小程序:
1. Java 脚本段。可以写 Java 代码,定义局部变量、编写语句等
   生成的代码在 servlet 中 `service` 方法体中。
   ```java
   <%
       // 可以写 Java 代码
   %>
   ```
2. 声明。声明全局变量、方法、类等
   生成的代码在 servlet 的类体中。
   ```java
   <%!
       // 声明全局变量
   %>
   ```
3. 输出表达式。可以输出变量或字面量
   生成的代码在 servlet 中的 `service` 方法体中，相当于 `out.print()` 输出
   ```java
   <%= 数值 %>
   ```

演示:

```java
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
```java
<%@include file="url"%>
```

**特点**:
1. 将页面进行了直接的替换
2. 静态包含只会生成一个源码文件，最终的内容全在 `_jspService` 方法体中(源码文件中)
3. 不能出现同名变量(因为会整合到同一个文件内)
4. 运行效率高一点点。耦合较高。

## include 动态包含

**格式**:
```java
<jsp:include page="url"></jsp:include>
```

**特点**:
1. 动态包含相当于方法的调用
2. 动态包含会生成多个源码文件
3. 可以定义同名变量
4. 效率高，耦合度低

**注**: 当动态包含不需要传递参数时， `include` 双标签之间不要有任何内容，包括换行和空格。

**传递参数**:
```java
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

```java
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
```java
<jsp:forward page="url"></jsp:forward>
```

**客户端**: 超链接