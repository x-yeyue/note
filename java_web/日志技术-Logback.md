# 常见的日志技术

- JUL: 这是 JavaSE 平台官方提供的官方日志框架。配置相对简单，不够灵活，性能较差。
- Log4j: 一个流行的日志框架，提供了灵活的配置选项，支持多种输出目标。
- Logback: 基于 Log4j 升级而来，提供了功能多的功能和配置选项，性能优于 Log4j。

***Slf4j***(Simple Logging Facade for Java): 简单日志门面，提供了一套日志操作的标准接口及抽象类，允许应用程序使用不同的底层日志框架。

# Logback 快速入门

## 准备工作

引入 logback 的依赖(springboot 项目中该依赖已传递)、配置文件 *logback.xml*。

```xml
pom.xml:

<dependency>
	<groupId>ch.qos.logback</groupId>
	<artifactId>logback-classic</artifactId>
	<version>1.4.11</version>
</dependency>
```

```xml
logback.xml

<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <!-- 控制台输出 -->
    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
            <!--格式化输出：%d表示日期，%thread表示线程名，%-5level：级别从左显示5个字符宽度  %logger{50}: 最长50个字符(超出.切割)  %msg：日志消息，%n是换行符 -->
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg%n</pattern>
        </encoder>
    </appender>
    
    <!-- 日志输出级别 -->
    <root level="debug">
        <appender-ref ref="STDOUT" />
    </root>
</configuration>
```

## 记录日志

定义日志记录对象 Logger，记录日志。

```java
public class LogTest {

    private static final Logger log = LoggerFactory.getLogger(LogTest.class);

    @Test
    public void testLog(){
        // System.out.println(LocalDateTime.now() + " : 开始计算...");
        log.debug("开始计算...");

        int sum = 0;
        int[] nums = {1, 5, 3, 2, 1, 4, 5, 4, 6, 7, 4, 34, 2, 23};
        for (int num : nums) {
            sum += num;
        }

        log.info("计算结果为: " + sum);
        // System.out.println("计算结果为: "+sum);
        // System.out.println(LocalDateTime.now() + "结束计算...");
        log.debug("结束计算...");
    }

}
```

控制台输出结果:

```text
2026-07-23 10:29:58.001 [main] DEBUG com.itheima.LogTest - 开始计算...
2026-07-23 10:29:58.005 [main] INFO  com.itheima.LogTest - 计算结果为: 101
2026-07-23 10:29:58.005 [main] DEBUG com.itheima.LogTest - 结束计算...
```

### 关闭控制台日志输出

将 *logback.xml* 文件中 `<root>` 的 `level` 属性设置为 `off`。

```xml
logback.xml:

<root level="off">
	<appender-ref ref="STDOUT" />
</root>
```

# 配置文件详解

配置文件名: *logback.xml*

该配置文件是对 logback 日志框架输出的日志进行控制的，可以配置输出的格式、位置及日志开关等。

常用的两种日志的位置: 控制台、系统文件。(也支持向数据库存储)

```xml
<!-- 控制台输出 -->
<appender name = "STDOUT" class = "ch.qos.logback.core.ConsoleAppender">...</appender>

<!-- 系统文件输出 -->
<appender name = "FILE" class = "ch.qos.logback.core.rolling.RollingFileAppender">...</appender>
```

开启日志(ALL)，关闭日志(OFF)

```xml
<root level="ALL">
	<appender-ref ref = "STDOUT" />  <!-- ref 中对应了 <appender> 的 name 属性 -->
	<appender-ref ref = "FILE" />
</root>
```

完整日志:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
	<!-- 控制台输出 -->
	<appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
		<encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
			<!--格式化输出：%d 表示日期，%thread 表示线程名，%-5level表示级别从左显示5个字符宽度，%logger显示日志记录器的名称， %msg表示日志消息，%n表示换行符 -->
			<pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50}-%msg%n</pattern>
		</encoder>
	</appender>

	<!-- 系统文件输出 -->
	<appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
		<rollingPolicy class="ch.qos.logback.core.rolling.SizeAndTimeBasedRollingPolicy">
			<!-- 日志文件输出的文件名, %i表示序号 -->
			<FileNamePattern>D:/tlias-%d{yyyy-MM-dd}-%i.log</FileNamePattern>
			<!-- 最多保留的历史日志文件数量 -->
			<MaxHistory>30</MaxHistory>
			<!-- 最大文件大小，超过这个大小会触发滚动到新文件，默认为 10MB -->
			<maxFileSize>10MB</maxFileSize>
		</rollingPolicy>

		<encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
			<!--格式化输出：%d 表示日期，%thread 表示线程名，%-5level表示级别从左显示5个字符宽度，%msg表示日志消息，%n表示换行符 -->
			<pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50}-%msg%n</pattern>
		</encoder>
	</appender>

	<!-- 日志输出级别 -->
	<root level="ALL">
		<appender-ref ref="STDOUT" />
		<appender-ref ref="FILE" />
	</root>
</configuration>

```

# 日志级别

日志级别指日志信息的类型，常见的日志级别(从低到高):

|   日志级别    | 说明                               |        记录方法        |
| :-------: | :------------------------------- | :----------------: |
| **trace** | 追踪，记录程序运行轨迹。                     | `log.trace("...")` |
| **debug** | 调试，记录程序调试过程中的信息，实际应用中一般将其视为最低级别。 | `log.debug("...")` |
| **info**  | 记录一般信息，描述程序运行的关键事件，如: 网络链接、io 操作 | `log.info("...")`  |
| **warn**  | 警告信息，记录潜在有害的情况。                  | `log.warn("...")`  |
| **error** | 错误信息。                            | `log.error("...")` |

```xml
<root level="info"> <!-- 大于等于配置的日志级别的日志才会输出 -->
	<appender-ref ref = "STDOUT" />
	<appender-ref ref = "FILE" />
</root>
```

# 简化操作

## 利用 @Slf4j 简化操作

```java
private static final Logger log = LoggerFactory.getLogger(LogTest.class);
```

当包含了 `lombok` 依赖后可以简化这一长串的繁琐定义。

```java
@Slf4j
public class LogTest {

    // private static final Logger log = LoggerFactory.getLogger(LogTest.class);

    @Test
    public void testLog(){
        // System.out.println(LocalDateTime.now() + " : 开始计算...");
        log.debug("开始计算...");

        int sum = 0;
        int[] nums = {1, 5, 3, 2, 1, 4, 5, 4, 6, 7, 4, 34, 2, 23};
        for (int num : nums) {
            sum += num;
        }

        log.info("计算结果为: " + sum);
        // System.out.println("计算结果为: "+sum);
        // System.out.println(LocalDateTime.now() + "结束计算...");
        log.debug("结束计算...");
    }

}
```

仅需在类前添加 `@Slf4j` 注解即可。

## 利用占位符简化日志输出

```java
log.info("计算结果为: " + sum);

// 多个同理
log.info("计算结果为: {}", sum);
```