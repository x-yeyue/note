# 快速入门

1. 创建 **javaEE** 项目
2. 定义一个类，实现 **Servlet** 接口
   `public class ServletDemo1 implements Servlet`
3. 实现接口中的抽象方法
4. 配置 Servelet
   ```java
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

