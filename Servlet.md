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

# 执行原理

1. 当服务器接受到客户端浏览器的请求后，会解析请求 $URL$ 路径，获取访问的 **Servlet** 的资源地址
2. 查找 **web.xml** 文件是否有对应的 `<url-pattern>` 标签体内容
3. 如果有，则找到其对应的 `<servlet-class>` 全类名
4. **tomcat** 会将字节码文件加载进内存，并且创还能其对象
5. 调用其方法

