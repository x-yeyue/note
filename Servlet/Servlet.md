# 快速入门

1. 创建 **javaEE** 项目
2. 定义一个类，实现 **Servlet** 接口
   `public class ServletDemo1 implements Servlet`
3. 实现接口中的抽象方法
4. 配置 Servelet
   ```xml
   // 在 web.xml 中配置
    <servlet> 
		<servlet-name>demo1</servlet-name> // name
	    <servlet-class>com.itheima.demo.ServletDemo1</servlet-class>  // name 对应的类
	</servlet>

    <servlet-mapping>
        <servlet-name>demo1</servlet-name> // 资源地址对应的 name 
        <url-pattern>/demo1</url-pattern> //访问的 资源地址
    </servlet-mapping>
   ```

## 执行原理

1. 当服务器接受到客户端浏览器的请求后，会解析请求 $URL$ 路径，获取访问的 **Servlet** 的资源地址
2. 查找 **web.xml** 文件是否有对应的 `<url-pattern>` 标签体内容
3. 如果有，则找到其对应的 `<servlet-class>` 全类名
4. **tomcat** 会将字节码文件加载进内存，并且创还能其对象
5. 调用其方法

## Servlet 的生命周期

1. 创建: 执行 `init` 方法，只执行一次
   -   Servlet 什么时候创建？
      - 默认情况下，第一次被访问时被创建
         - 配置 Servlet 的创建时机(`web.xml`)
           ![Servlet生命周期init方法的执行时机](Servlet生命周期init方法的执行时机.png "Servlet生命周期init方法的执行时机.png")
	- Servlet 的 `init` 方法，只执行一次，说明一个 Servlet 在内存中只存在一个对象，**Servlet 是单例的**
		- 多个用户同时访问时，可能存在线程安全问题
		- 解决: 尽量不要在 Servlet 中定义成员变量。即使创建了成员变量，也不要修改值。
2. 提供服务: 执行 `service` 方法，执行多次
   每次访问 Servlet 时，`service` 方法都会被调用一次。
3. 销毁: 执行 `destroy` 方法，只执行一次
   Servlet 被销毁时执行。服务器关闭时，Servlet 被销毁。
   只有服务器正常关闭时，才会执行 `destroy` 方法。
   具体的，`destroy` 方法在 Servlet 被销毁之前执行，一般用于释放资源。

## Servlet3.0

支持注解配置。可以不用 `web.xml` 了。

**步骤**:
1. 创建 `JavaEE` 项目，选择 Servlet 的版本 3.0 以上，可以不创建 `web.xml`
2. 定义一个类，实现 Servlet 接口
3. 复写方法
4. 在类上使用 `@WebServlet` 注解，进行配置
   `@WebServlet("/demo")`

## Servlet 的体系结构

```text
Servlet -- 接口
	|
	|
GenericServlet -- 抽象类
	|
	|
HttpServlet -- 抽象类
```

### Servlet 接口

定义了，完整的生命周期方法。

1. `init(ServletConfig config)`: 初始化方法
   在 Servlet 被创建的时候执行，只执行一次，用于初始化 Servlet 的资源。
2. `service(ServletRequest req, ServletResponse res)`: 提供服务方法
   每一次 Servlet 被访问都会执行一次，用于处理请求和响应。
3. `destroy()`: 销毁方法
   在 Servlet 被正确销毁的时候执行，用于释放 Servlet 的资源。
4. `getServletConfig()`: 获取 ServletConfig 对象
   用于获取 Servlet 的配置信息，如参数值、初始化参数等。
5. `getServletInfo()`: 获取 Servlet 的信息
   用于获取 Servlet 的名称、版本号等信息。

**痛点**: 每次都要重写所有方法，太繁琐。

### GenericServlet 抽象类

解决了 `Servlet` 接口的痛点。

实现了 `Servlet` 接口，并完成以下任务:
1. 将非核心的方法，默认空实现。
2. 将 `service()` 方法声明为 `abstract` 抽象方法。
   只需继承 `GenericServlet`，重写 `Service()` 方法即可，无需关注其他。

**痛点**: 由于方法中拿到的为 `ServletRequest`，需要转换为 `HttpServletRequest` 才能拿到 会话、请求头等信息，较麻烦。

### HttpServlet 抽象类

解决了 `GenericServlet` 的痛点。

`HttpServlet` 继承自 `GenericServlet`，针对 **HTTP协议** 做了封装:
1. 重写 `service()` 方法: 自动解析了 HTTP 请求方法。
2. 自动进行路由分发: 实现了 *分发逻辑* -- 如果是 GET 请求就调用 `doGet()` 方法;如果是 POST 请求，就调用 `doPost()` 方法。

继承 `HttpServlet` 后仅需重写 `doGet` 或 `doPost` 方法。

## Servlet 相关配置

1. `urlpartten`: Servlet 访问路径
   一个 Servlet 可以定义多个访问路径: `@WebServlet({"/demo3", "/dd4"})`
   
   路径定义规则:
   - `/xxx`
   - `/xxx/xxx`
   - \*.do

## request 功能

### 获取请求消息数据

#### 获取请求行数据
```text
GET /ServletDemo/demo1?name=x_yeyue HTTP/1.1
```

获取 **请求方式**: `GET`
```java
String getMethod()
```

获取 **虚拟目录**: `/ServletDemo`
```java
String getContextPath()
```

获取 **Servlet 路径**: `/demo1`
```java
String getServletPath()
```

获取 **get 方法请求参数**: `name=x_yeyue`
```java
String getQueryString()
```

获取 **请求 URI**: `/ServletDemo/demo1`
```java
String getRequestURI()
```

获取 **请求 URL**: `http://localhost/ServletDemo/demo1`
```java
StringBuffer getRequestURL()
```

获取 **版本协议**: `HTTP/1.1`
```java
String getProtocol()
```

获取 **客户机的 IP 地址**:
```java
String getRemoteAddr()
```

#### 获取请求头数据

```java
// 获取所有的请求头名称
Enumeration<String> getHeaderNames()

// 通过请求头的名字 获取请求头的值
String getHeader(String name)

// 获取所有请求头名称  
Enumeration<String> headerNames = req.getHeaderNames();  
// 遍历所有请求头名称  
while(headerNames.hasMoreElements()){  
    String name = headerNames.nextElement();  
    String value = req.getHeader(name);  
    System.out.println(name + " -> " + value);  
}
```

#### 获取请求体数据

> 请求体只有 POST 请求方式，才有请求体，在请求体中封装了 POST 请求的请求参数。

**步骤**: 
1. 获取流对象
   ```java
    // 获取字符输入流，只能操作字符数据
	BufferedReader getReader()
   
	// 获取字节输入流，可以操作所有类型数据
	ServletInputStream getInputStream()
   
   
	BufferedReader reader = req.getReader();  
	String line = null;  
	while((line = reader.readLine()) != null){  
		System.out.println(line);  
	}
}
   ```
2. 再从流对象中拿数据

### 其他功能

#### 获取请求参数通用方法

```java
/*
不论 GET 还是 POST 
*/

// 根据参数名称获取参数值  username=x_yeyue&password=123
String getParameter(String name)

// 根据参数名称获取参数值的数组  hobby=xx&hobby=game
String[] getParameterValues(String name)

// 获取所有请求的参数名称
Enumeration<String> getParameterNames()

// 获取所有参数的 map 集合
Map<String, String[]> getParameterMap()
```

##### 中文乱码问题

- get 方法: tomcat8 已经将 get 方法乱码问题解决。
- post 方法: 在获取参数前，设置 request 的编码 `request.setCharacterEncoding("utf-8");`

#### 请求转发

**步骤**
1. 通过 `request` 对象获取请求转发器对象
   ```java
   RequestDispatcher getRequestDispatcher(String path)
   ```
2. 使用 `RequestDispatcher` 对象来进行转发
   ```java
   forward(ServletRequest request, ServletResponse response)
   ```

```java
@Override  
protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {  
    System.out.println("demo444444");  
  
    // 转发到 demo5  
    req.getRequestDispatcher("/requestDemo5").forward(req, resp);  
    System.out.println("demo444444 转发到 demo5");  
  
    /*  
    RequestDispatcher dispatcher = req.getRequestDispatcher("/requestDemo5");   
    dispatcher.forward(req, resp);     
    */
}
```

**特点**:
1. 浏览器地址栏路径不发生变化
2. 只能转发到当前服务器内部资源中
3. 转发是一次请求

#### 共享数据

> **域对象**: 一个有作用范围的对象，可以在范围内共享数据

`request`域: 一次请求的范围，一般用于请求转发的多个资源中共享数据

```java
// 存储数据
void setAttribute(String name, Object obj)

// 通过键获取值
Object getAttitude(String name)

// 通过键移除键值对
void removeAttribute(String name)
```

#### 获取 ServletContext

```java
ServletContext getServletContext()
```


