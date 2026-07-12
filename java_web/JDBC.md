# 依赖

```xml
<dependencies>  
    <dependency>
		<groupId>com.mysql</groupId>
		<artifactId>mysql-connector-j</artifactId>
		<version>8.0.33</version>
	</dependency>
    <dependency>
        <groupId>org.junit.jupiter</groupId>  
        <artifactId>junit-jupiter-engine</artifactId>  
        <version>5.9.3</version>  
        <scope>test</scope>  
    </dependency>
    <dependency>
        <groupId>org.projectlombok</groupId>  
        <artifactId>lombok</artifactId>  
        <version>1.18.30</version>  
    </dependency>
</dependencies>
```

# 入门程序

数据更新

```java
public class JdpcTest {
    /**
     * JDPC 入门程序
     */
    @Test
    public void testUpdate() throws Exception {
        // 1. 注册驱动
        Class.forName("com.mysql.cj.jdbc.Driver");
        
        // 2. 获取连接
        String url = "jdbc:mysql://localhost:3306/web01";
        String username = "root";
        String password = "051011";
        Connection connection = DriverManager.getConnection(url, username, password);
        
        // 3. 获取 SQL 语句的执行对象
        Statement statement = connection.createStatement();
        
        // 4. 执行 SQL
        int i = statement.executeUpdate("update user set age = 25 where id = 1");
        System.out.println("sql 语句执行完毕，影响的行数为：" + i);
        
        // 5. 释放资源
        statement.close();
        connection.close();
    }
}
```

数据查询

```java
@Test
public void testSelect(){
	// 数据库连接信息（请根据实际环境修改）
	String url = "jdbc:mysql://localhost:3306/web01";
	String dbUser = "root";
	String dbPassword = "051011";

	// 要执行的 SQL
	String sql = "SELECT id, username, password, name, age FROM user WHERE username = ? AND password = ?";

	// JDBC 对象
	Connection conn = null;
	PreparedStatement pstmt = null;
	ResultSet rs = null; // 封装查询返回的结果

	try {
		// 1. 加载驱动（可省略，但显式声明更清晰）
		Class.forName("com.mysql.cj.jdbc.Driver");

		// 2. 获取连接
		conn = DriverManager.getConnection(url, dbUser, dbPassword);

		// 3. 创建预编译语句
		pstmt = conn.prepareStatement(sql);
		pstmt.setString(1, "daqiao");
		pstmt.setString(2, "123456");

		// 4. 执行查询
		rs = pstmt.executeQuery();

		// 5. 遍历结果集，封装到 User 对象并打印
		while (rs.next()) {
			User user = new User();
			user.setId(rs.getInt("id"));
			user.setUsername(rs.getString("username"));
			user.setPassword(rs.getString("password"));
			user.setName(rs.getString("name"));
			user.setAge(rs.getInt("age"));

			// 输出到控制台（使用 Lombok 自动生成的 toString）
			System.out.println(user);
		}

	} catch (ClassNotFoundException e) {
		System.err.println("数据库驱动未找到：" + e.getMessage());
	} catch (SQLException e) {
		System.err.println("数据库操作异常：" + e.getMessage());
	} finally {
		// 6. 释放资源（逆序关闭）
		try {
			if (rs != null) rs.close();
			if (pstmt != null) pstmt.close();
			if (conn != null) conn.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}
}
```


`ResultSet` 结果集对象: `ResultSet rs = statement.executeQuery()`
- `next()` 将光标从当前位置先后移动一行，并判断当前行是否为有效行，返回值为 `bollean`。
- `getXxx(...)` 获取数据，可根据列的编号或列名获取。

预编译 SQL 

```java
PreparedStatement ps = conn.prepareStatement("select * from user where username = ? and password = ?);
ps.setString(1, "linchong");
ps.setString(2, "123456");
ResultSet resultSet = ps.executeQuery();
```