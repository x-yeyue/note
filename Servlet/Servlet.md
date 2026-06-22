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
         - 配置 Servlet 的创建时机
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

