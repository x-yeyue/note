## 导入 SpringBoot 模块

![新建springboot1.png](./img/新建springboot1.png)

![新建springboot1.png](./img/新建springboot2.png)

![新建springboot1.png](./img/新建springboot3.png)

![新建springboot1.png](./img/新建springboot4.png)

## 基础例子

**需求:**

![](img/basic_requeir.png)

浏览器向服务器的 `hello` 地址发起请求，服务器获取请求中的 `name` 值并按照图中格式输出，并返回给浏览器。

**实现:**

![](img/basic_requeir_code.png)

在如图目录中创建 `HelloController` 类文件，并实现如图代码。

在 `SpringbootWebQuickstartApplication` 中运行后，实现该效果。如图:

![](img/basic_requeir_achieve.png)

## 获取请求参数

`/dept?id=14`

### HttpServletRequest

```java
/**
 * 第一种方式 HttpServletRequest
 */
@DeleteMapping("/depts")
public Result delete(HttpServletRequest request){
	String idStr = request.getParameter("id");
	int id = Integer.parseInt(idStr);
	System.out.println("根据 ID 删除部门: " + id);
	return Result.success();
}
```

### @RequestParam

```java
/**
 * 第二种方式 @RequestParam
 * 注: 一旦设置了 @RequestParam 注解，则指定参数必须传递，否则会报错。(默认 required = true)
 */
@DeleteMapping("/depts")
public Result delete(@RequestParam(value = "id", required = false) Integer deptId){
	System.out.println("根据 ID 删除部门: " + deptId);
	return Result.success();
}
```

#### 设置参数默认值

```java
@GetMapping("/emps")
public Result page(@RequestParam(defaultValue = "1") Integer page, @RequestParam(defaultValue = "10") Integer pageSize){
        log.info("分页查询: {}, {}", page, pageSize);
        PageResult<Emp> pageResult = empService.page(page, pageSize);
        return Result.success(pageResult);
    }
```


### 省略 @RequestParam (推荐)

```java
/**
 * 第三种方式，省略 @RequestParam
 * 当请求参数名与形参变量名相同，直接定义方法形参即可
 * /depts?id=15
 */
@DeleteMapping("/depts")
public Result delete(Integer id){
	System.out.println("根据 ID 删除部门: " + id);
	return Result.success();
}
```

### @RequestBody (接收 json 数据)

> 接收 json 格式的请求参数: `POST /depts {"name": "教研室"}`

json 格式的参数，通常会使用一个实体对象进行接收。

**规则**: json 数据的 *键名* 与方法形参 *对象的属性名* 相同，并需要使用 `@RequestBody` 注解标识。

```java
@PostMapping("/depts")
public Result add(@RequestBody Dept dept){
	System.out.println("新增部门: " + dept);
	return Result.success();
}
```

```java
public class Dept {
    private Integer id;
    private String name;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
```

### 获取日期时间

请求路径中的日期格式可能会有很多，例如: `yyyy-MM-dd`、`yyyy/MM/dd`、`yyyy年MM月dd日` 等等。

通过 `@DateTimeFormat` 注解，指定 `pattern` 参数来确定日期格式，该注解会将日期按照该格式解析后传递给后面的参数。

```java
/**
 * 请求路径: /emps?name=张&gender=1&begin=2007-09-01&end=2022-09-01&page=1&pageSize=10
*/
@GetMapping
public Result page(@RequestParam(defaultValue = "1") Integer page,
				   @RequestParam(defaultValue = "10") Integer pageSize,
				   String name,
				   Integer gender,
				   @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate begin,
				   @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate end){
	log.info("分页查询: page-{}, pageSize-{}, name-{}, gender-{}, begin-{}, end-{}", page, pageSize,  name, gender, begin, end);
	PageResult<Emp> pageResult = empService.page(page, pageSize, name, gender, begin, end);
	return Result.success(pageResult);
}
```

### @PathVariable(接收 路径参数)

> 接收路径参数: `GET /depts/1`

路径参数: 通过 URL 直接传递参数，使用 `{...}` 来标识该路径参数，需要使用 `@PathVariable` 获取路径参数。

```java
@GetMapping("/depts/{id}")
public Result getInfo(@PathVariable("id") Integer deptId){
	System.out.println("根据  ID 查询部门: " + deptId);
	return Result.success();
}
```

当路径参数和形参一致时，简略写法:

```java
@GetMapping("/depts/{id}")
public Result getInfo(@PathVariable Integer id){
	System.out.println("根据  ID 查询部门: " + id);
	return Result.success();
}
```

## 接收数组

`/emps?ids=1,2,3`

1. 通过数组接收
   ```java
   @DeleteMapping  
	public Result delete(Integer[] ids){  
	    log.info("根据ID删除员工: {}", ids);  
	    return Result.success();  
	}
   ```
2. 通过集合接收
   ```java
   // 将接收到的参数封装到集合等复杂类型中必须写 @RequestParam 注解
   @DeleteMapping
    public Result delete(@RequestParam List<Integer> ids){
        log.info("根据ID删除员工: {}", ids);
        return Result.success();

    }
   ```

## 请求路径简化

当请求路径中有公共前缀时，可以在类前通过 `@RequestMapping` 注解标注公共部分。

原:

```java
@RestController
public class DeptController {

    @Autowired
    private DeptService deptService;

    @GetMapping("/depts")
    public Result list(){
        System.out.println("查询全部的部门数据");
        List<Dept> deptList = deptService.findAll();
        return Result.success(deptList);
    }

    @DeleteMapping("/depts")
    public Result delete(Integer id){
        System.out.println("根据 ID 删除部门: " + id);
        deptService.deleteById(id);
        return Result.success();
    }

    @PostMapping("/depts")
    public Result add(@RequestBody Dept dept){
        System.out.println("新增部门: " + dept);
        deptService.add(dept);
        return Result.success();
    }

    @GetMapping("/depts/{id}")
    public Result getInfo(@PathVariable Integer id){
        System.out.println("根据  ID 查询部门: " + id);
        Dept dept = deptService.getById(id);
        return Result.success(dept);
    }

    @PutMapping("/depts")
    public Result update(@RequestBody Dept dept){
        System.out.println("根据 id 更新部门信息: " + dept);
        deptService.update(dept);
        return Result.success();
    }
}
```

简化:

```java
@RequestMapping("/depts")
@RestController
public class DeptController {

    @Autowired
    private DeptService deptService;

    @GetMapping
    public Result list(){
        System.out.println("查询全部的部门数据");
        List<Dept> deptList = deptService.findAll();
        return Result.success(deptList);
    }

    @DeleteMapping
    public Result delete(Integer id){
        System.out.println("根据 ID 删除部门: " + id);
        deptService.deleteById(id);
        return Result.success();
    }

    @PostMapping
    public Result add(@RequestBody Dept dept){
        System.out.println("新增部门: " + dept);
        deptService.add(dept);
        return Result.success();
    }

    @GetMapping("/{id}")
    public Result getInfo(@PathVariable Integer id){
        System.out.println("根据  ID 查询部门: " + id);
        Dept dept = deptService.getById(id);
        return Result.success(dept);
    }

    @PutMapping
    public Result update(@RequestBody Dept dept){
        System.out.println("根据 id 更新部门信息: " + dept);
        deptService.update(dept);
        return Result.success();
    }
}
```

# 分页查询

## 原始方案

```java
// =============== Mapper 层 ===============

/**
 * 查询总记录数
 */
@Select("Select count(*) from emp ...")
public Long count();

/**
 * 分页查询
 */
@Select("Select emp.*, dept.name deptName from emp ... limit #{start}, #{pageSize} ")
public List<Emp> list(Integer start, Integer pageSize);

// =============== Service 层 ===============

public PageResult<Emp> page(Integer page, Integer pageSize) {
	log.info("获取 PageResult.");
	Integer start = (page - 1) * pageSize;
	Long total = empMapper.count(); 
	log.info("用户信息条数: {}", total);
	List<Emp> empList = empMapper.list(start, pageSize);
	return new PageResult<Emp> (total, empList);
}
```

## PageHelper 插件

![[PageHelper]]

## 条件分页查询

### 基本实现

请求参数: `emps?name=阮&gender=1&begin=2007-09-01&end=2022-09-01&page=1&pageSize=10`

controller层:

```java
public Result page(@RequestParam(defaultValue = "1") Integer page,
                       @RequestParam(defaultValue = "10") Integer pageSize,
                       String name,
                       Integer gender,
                       @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate begin,
                       @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate end){
        log.info("分页查询: page-{}, pageSize-{}, name-{}, gender-{}, begin-{}, end-{}", page, pageSize,  name, gender, begin, end);
        PageResult<Emp> pageResult = empService.page(page, pageSize, name, gender, begin, end);
        return Result.success(pageResult);
    }
```

- `@DateTimeFormat()` 注解参考 [获取日期时间](#获取日期时间)。

service 层及 mapper 层见 [多参数查询](PageHelper.md#多参数查询) 部分的图片。

SQL 代码:

```sql
select emp.*, dept.name from emp left join dept on emp.dept_id = dept.id
	where
		emp.name like concat('%', #{name}, '%') and
		emp.gender = #{gender} and
		emp.entry_date
	between #{begin} and #{end}
	order by emp.update_time desc
```

`concat()` 函数参考 [MySQL字符串拼接](../class/mysql/mysql_基础操作.md#字符串拼接) 。

如果直接使用 `emp.name like '%#{name}%'` 替换 `concat()` 函数，在编译后，会形成 `emp.name like '%?%'`，由于外部的引号，该位置的就会变成一个字符串，就会导致参数无法传递，从而报错。

### 程序优化

#### 请求参数接收优化

如果 controller 方法的 *参数较多* ，且未来可能继续增加，这会使得方法签名变得复杂难以维护，此时可以考虑将多个请求参数 **封装为一个对象**。

```java
@Data
public class EmpQueryParam {
    private Integer page = 1; // 当前页码
    private Integer pageSize = 10; // 每页记录数
    private String name; // 员工姓名
    private Integer gender; // 员工性别
    @DateTimeFormat(pattern = "yyyy-MM-dd")
    private LocalDate begin; // 查询入职日期起始时间
    @DateTimeFormat(pattern = "yyyy-MM-dd")
    private LocalDate end; // 入职日期日期结束时间
}
```

```java
// ================== controller ==================
@GetMapping  
public Result page(EmpQueryParam empQueryParam) {  
    log.info("分页查询: {}", empQueryParam);  
    PageResult<Emp> pageResult = empService.page(empQueryParam);  
    return Result.success(pageResult);  
}

// ================== service ==================
public PageResult<Emp> page(EmpQueryParam empQueryParam) {  
    // 设置分页参数  
    PageHelper.startPage(empQueryParam.getPage(), empQueryParam.getPageSize());  
    // 调用 Mapper 接口方法  
    List<Emp> empList = empMapper.list(empQueryParam);  
    // 解析并封装结果  
    Page<Emp> p = (Page<Emp>) empList;  
    return new PageResult<Emp>(p.getTotal(), p.getResult());  
}

// ================== mapper ==================
public List<Emp> list(EmpQueryParam empQueryParam);
```

**注:** mapper层使用了 [XML 映射配置(SQL)](Mybatis.md#XML%20映射配置(SQL))。

```sql
select emp.*, dept.name from emp left join dept on emp.dept_id = dept.id  
    where        
	    emp.name like concat('%', #{name}, '%') and        
	    emp.gender = #{gender} and        
	    emp.entry_date between #{begin} and #{end}    
	order by emp.update_time desc
```

#### 动态 SQL 优化

当有多个传递参数时，可能有部分传递参数为 `null` 即为传递，当 SQL 被写死，就会出现报错，通过 Mybatis 的 [动态 SQL](Mybatis.md#动态%20SQL) 进行优化，就能解决传递参数缺失造成的报错问题。

```xml
<select id = "list" resultType = "com.itheima.pojo.Emp">  
    select emp.*, dept.name from emp left join dept on emp.dept_id = dept.id  
        <where>  
            <if test="name != null and name != ''">  
                emp.name like concat('%', #{name}, '%')  
            </if>  
            <if test="gender != null">  
                and emp.gender = #{gender}  
            </if>  
            <if test="begin != null  and end != null">  
                and emp.entry_date between #{begin} and #{end}  
            </if>  
        </where>  
        order by emp.update_time desc  
</select>
```

# 启用数据库自动产生的主键

当有些表的主键设置了*自增*属性，那么在插入新的记录的时候就需要尽量避开为这个字段赋值。

那么如果在插入记录后，就要利用这个新增 记录的主键呢？

选择用单独的查询语句来查询吗？

可以利用 Mybatis 的设置快速获取。

## 注解映射

   ```java
   @Options(useGeneratedKeys = true, keyProperty = "id")
   @Insert("insert into ......")
   void insert(Emp emp);
   ```

## XML 映射

   ```java
   void insert(Emp emp);
   
   
   ------------ xml ------------
    <insert id="insert" useGeneratedKeys="true" keyProperty="id">  
	    insert into ......
	</insert>
   ```

# 事务管理

## 通过注解操作

概念见 [事务管理](../class/mysql/mysql_基础操作.md#事务管理)。

- 注解: `@Transactional`
- 作用: 将当前方法交给 spring 进行事务管理，方法执行前，开启事务；成功执行完毕、提交事务；出现异常，回滚事务。
- 位置: 业务层(service) 的方法上、类上、接口上。

```java
@Transactional  
@Override  
public void save(Emp emp) {  
    // 保存员工基本信息  
    emp.setCreateTime(LocalDateTime.now());  
    emp.setUpdateTime(LocalDateTime.now());  
    empMapper.insert(emp);  
  
    // 保存员工工作经历信息  
    List<EmpExpr> exprList = emp.getExprList();  
    if(!exprList.isEmpty()){  
        // 遍历集合，为每项 empExpr 的 empId 赋值  
        exprList.forEach(empExpr -> {  
            empExpr.setEmpId(emp.getId());  
        });  
  
        empExprMapper.insertBatch(exprList);  
    }  
}
```

> [!tips]
> 推荐将 `@Transactional` 注解放在有多个数据库操作的方法上。

## 设置事务管理的日志信息

```yml
# 配置事务管理日志级别  
logging:  
  level:  
    org.springframework.jdbc.support.JdbcTransactionManager: debug
```


## 事务进阶

### 控制回滚事务类型 rollbackFor

`@Transactional` 的 `rollbackFor` 属性用于控制出现何种异常类型，回滚事务。

```java
@Transactional(rollbackFor = {Exception.class}) // 所有异常都回滚
@Override  
public void save(Emp emp) {  
    // 保存员工基本信息  
    emp.setCreateTime(LocalDateTime.now());  
    emp.setUpdateTime(LocalDateTime.now());  
    empMapper.insert(emp);  
  
    // 保存员工工作经历信息  
    List<EmpExpr> exprList = emp.getExprList();  
    if(!exprList.isEmpty()){  
        // 遍历集合，为每项 empExpr 的 empId 赋值  
        exprList.forEach(empExpr -> {  
            empExpr.setEmpId(emp.getId());  
        });  
  
        empExprMapper.insertBatch(exprList);  
    }  
}
```

> [!tips]
> 默认情况下仅会在出现 运行时异常 `RuntimeException` 事务才会回滚。

### 事务的传播行为 propagation

`propagation` 属性用于设置事务传播行为: 当一个事务方法被另一个事务方法调用时，这个事务方法应该如何进行事务控制。

```java
@Transacational
public void a(){
	// ...
	userService.b();
	// ...
}

------

@Transacational(propagation = Propagation.REQUIRED)
public void b(){
	// ...
}
```

|      属性值      |                含义                 |
| :-----------: | :-------------------------------: |
|   REQUIRED    |      \[默认值]需要事务，有则加入，无则创建新事物      |
| REQUIRES_NEW  |        需要新事务，无论有无，总是创建新事务         |
|   SUPPORTS    |       支持事务，有则加入，无则在无事务状态运行        |
| NOT_SUPPORTED | 不支持事务，在无事务状态运行，如果当前存在已有事务，则挂起当前事务 |
|   MANDATORY   |            必须有事务，否则抛异常            |
|     NEVER     |            必须没事务，否则抛异常            |

#### 样例

需求: 在新增员工信息时，无论是成功还是失败，都要记录操作日志。

```java
// ============== EmpServiceImpl ==============
@Transactional(rollbackFor = {Exception.class}) // 事务管理(设置所有异常都回滚)  
@Override  
public void save(Emp emp) throws Exception {  
    try {  
        // 保存员工基本信息  
        emp.setCreateTime(LocalDateTime.now());  
        emp.setUpdateTime(LocalDateTime.now());  
        empMapper.insert(emp);  
  
        Integer i = 1/0;  
        
        List<EmpExpr> exprList = emp.getExprList();  
        if(!exprList.isEmpty()){  
            // 遍历集合，为每项 empExpr 的 empId 赋值  
            exprList.forEach(empExpr -> {  
                empExpr.setEmpId(emp.getId());  
            });  
  
            empExprMapper.insertBatch(exprList);  
        }  
    } finally {  // 记录操作日志
        EmpLog empLog = new EmpLog(null, LocalDateTime.now(), "新增员工: " + emp);  
        empLogService.insertLog(empLog);
    }  
}

// ============== EmpLogServiceImpl ==============
@Override  
public void insertLog(EmpLog empLog) {  
    empLogMapper.insert(empLog);  
}
```

当调用后，由于 `i / 0` 发生了错误，即使将记录操作日志的行为放到了 `finally` 块中，其代码的确执行了，但由于该方法发生了错误，最终会被 *事务回滚*，导致操作日志没有成功写入数据库。 

解决方案: 在 `EmpLogServiceImpl` 的 `insertLog` 方法添加事务管理，同时设置 `propagation` 属性为 `REQUIRES_NEW`，在一个新创建的事务中运行。

```java
// ============== EmpServiceImpl ==============
@Transactional(rollbackFor = {Exception.class}) // 事务管理(设置所有异常都回滚)  
@Override  
public void save(Emp emp) throws Exception {  
    try {  
        // 保存员工基本信息  
        emp.setCreateTime(LocalDateTime.now());  
        emp.setUpdateTime(LocalDateTime.now());  
        empMapper.insert(emp);  
  
        Integer i = 1/0;  
        
        List<EmpExpr> exprList = emp.getExprList();  
        if(!exprList.isEmpty()){  
            // 遍历集合，为每项 empExpr 的 empId 赋值  
            exprList.forEach(empExpr -> {  
                empExpr.setEmpId(emp.getId());  
            });  
  
            empExprMapper.insertBatch(exprList);  
        }  
    } finally {  // 记录操作日志
        EmpLog empLog = new EmpLog(null, LocalDateTime.now(), "新增员工: " + emp);  
        empLogService.insertLog(empLog);
    }  
}

// ============== EmpLogServiceImpl ==============
@Transactional(propagation = Propagation.REQUIRES_NEW) // 在新建的事务中运行  
@Override  
public void insertLog(EmpLog empLog) {  
    empLogMapper.insert(empLog);  
}
```

# 文件上传

## 基本实现

前端:

```html
<!-- 
要求:
1. 表单提交方式 method 必须为 post (提交文件可能很大，get 允许空间较小)
2. form 标签需要添加 enctype 属性，并设置 属性为 "multipart/form-data"
   当不设置 enctype，上传到服务器的为文件名，设置后才为文件本体。
3. 必须存在一个 type 为 file 的 input 标签。
-->
<form action="/upload" method="post" enctype="multipart/form-data">  
    姓名: <input type="text" name="name"><br>  
    年龄: <input type="text" name="age"><br>  
    头像: <input type="file" name="file"><br>  
    <input type="submit" value="提交">  
</form>
```

后端:

```java
@PostMapping("/upload")  
// 参数需要和前端的 input 标签的 name 属性值一致，否则需要使用 @RequestParam 注解指定
public Result upload(String name, Integer age, MultipartFile file) {  
    log.info("接收参数: {}, {}, {}", name, age, file);  
    return Result.success();  
}
```

## 本地存储

```java
@PostMapping("/upload")  
public Result upload(String name, Integer age, MultipartFile file) throws IOException {  
    log.info("接收参数: {}, {}, {}", name, age, file);  
    // 获取原始文件名  
    String originalFilename = file.getOriginalFilename();  
  
    // 新文件名  
    String extension = originalFilename.substring(originalFilename.lastIndexOf('.'));  
    String newFileName = UUID.randomUUID().toString() + "." + extension;  
  
    //保存文件  
    file.transferTo(new File("D:/File/AI_web/web-ai-project02/tlias-web-management/src/main/resources/imgs/" + newFileName));  
  
    return Result.success();  
}
```

### 上传文件大小限制设置

```yml
# application.yml

spring:  
  application:  
    name: tlias-web-management  
  datasource:  
    url: jdbc:mysql://localhost:3306/tlias  
    driver-class-name: com.mysql.cj.jdbc.Driver  
    username: root  
    password: "123456"  
  servlet:  
    multipart:  
      # 最大单个文件大小  
      max-file-size: 10MB  
      # 最大请求大小  
      max-request-size: 100MB
```

## 阿里云OSS

```java
@Component  
public class AliyunOSSOperator {  
    // 方式一: @Value属性一个属性一个属性的注入  
    // @Value("${aliyun.oss.endpoint}")    
    // private String endpoint;    
    //    
    // @Value("${aliyun.oss.bucketName}")    
    // private String bucketName;    
    //    
    // @Value("${aliyun.oss.region}")    
    // private String region;  
    
    // 方式二: 通过类封装    
    @Autowired  
    AliyunOssProperties aliyunOssProperties;  
  
  
    public String upload(byte[] content, String originalFilename) throws Exception {  
        String endpoint = aliyunOssProperties.getEndpoint();  
        String region = aliyunOssProperties.getRegion();  
        String bucketName = aliyunOssProperties.getBucketName();  
  
        // 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。  
        EnvironmentVariableCredentialsProvider credentialsProvider = CredentialsProviderFactory.newEnvironmentVariableCredentialsProvider();  
  
        // 填写Object完整路径，例如202406/1.png。Object完整路径中不能包含Bucket名称。  
        //获取当前系统日期的字符串,格式为 yyyy/MM        
        String dir = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy/MM"));  
        //生成一个新的不重复的文件名  
        String newFileName = UUID.randomUUID() + originalFilename.substring(originalFilename.lastIndexOf("."));  
        String objectName = dir + "/" + newFileName;  
  
        // 创建OSSClient实例。  
        ClientBuilderConfiguration clientBuilderConfiguration = new ClientBuilderConfiguration();  
        clientBuilderConfiguration.setSignatureVersion(SignVersion.V4);  
        OSS ossClient = OSSClientBuilder.create()  
                .endpoint(endpoint)  
                .credentialsProvider(credentialsProvider)  
                .clientConfiguration(clientBuilderConfiguration)  
                .region(region)  
                .build();  
  
        try {  
            ossClient.putObject(bucketName, objectName, new ByteArrayInputStream(content));  
        } finally {  
            ossClient.shutdown();  
        }  
  
        return endpoint.split("//")[0] + "//" + bucketName + "." + endpoint.split("//")[1] + "/" + objectName;  
    }  
  
}
```

### 利用 @ConfigurationProperties 快速注入配置

```java
@Data  
@Component  
@ConfigurationProperties(prefix = "aliyun.oss")  
public class AliyunOssProperties {  
    private String endpoint;  
    private String bucketName;  
    private String region;  
}
```

```yml
# application.yml
# 阿里云oss 配置  
aliyun:  
  oss:  
    endpoint: "https://oss-cn-beijing.aliyuncs.com"  
    bucketName: "java-ai-x-yeyue"  
    region: "cn-beijing"
```

# 全局异常处理器

由于后端未处理异常，当程序发生异常后，异常层层向上抛，最终由框架拿到后扔给前端一个无法识别的异常，导致前端正确提示错误。

![springboot_全局异常处理器1](img/springboot_全局异常处理器1.png)

按照接口文档，应当返回如下格式:

```json
{
	"code": 2,
	"msg": "error",
	"data": []
}
```

解决方案: 添加 全局异常处理器，将 Controller 层抛出的异常拦截，并返回给前端正确的 json 格式信息。

```java
/**
 * 全局异常处理器
 */
@Slf4j
@RestControllerAdvice // 标识全局异常处理器
public class GlobalExceptionHandler {

    @ExceptionHandler // 标识异常处理方法，默认捕获所有异常
    public Result handleException(Exception e) {
        log.error("程序出错了", e);
        return Result.error("出错了请联系管理员~");
    }
}
```

## 输出具体信息

以添加员工时，电话号码重复为例，终端会报以下信息:

```
org.springframework.dao.DuplicateKeyException: 
### Error updating database.  Cause: java.sql.SQLIntegrityConstraintViolationException: Duplicate entry '18809091234' for key 'emp.phone'
### The error may exist in com/itheima/mapper/EmpMapper.xml
### The error may involve defaultParameterMap
### The error occurred while setting parameters
### SQL: insert into emp(username, name, gender, phone, job, salary, image, entry_date, dept_id, create_time, update_time)         values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
### Cause: java.sql.SQLIntegrityConstraintViolationException: Duplicate entry '18809091234' for key 'emp.phone'
; Duplicate entry '18809091234' for key 'emp.phone'
```

可以发现是 `DuplicateKeyException` 错误，后面有具体的报错信息，代码如下:

```java
/**
 * 全局异常处理器
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler
    public Result handleException(Exception e) {
        log.error("程序出错了", e);
        return Result.error("出错了请联系管理员~");
    }

    @ExceptionHandler
    public Result handleDuplicateKeyException(DuplicateKeyException e) {
        log.error("程序出错了", e);
        String message = e.getMessage(); // 获取报错信息
        int i = message.indexOf("Duplicate entry"); // 检索关键词
        String errMsg = message.substring(i); // 截取信息
        String[] arr = errMsg.split(" "); // 利用 " " 进行拆分
        return Result.error(arr[2] + " 已存在"); // 获取电话号码并返回数据
    }
}
```

> [!tips]
> 系统会根据异常的继承关系，从下往上匹配。
> 
> 当遇到 `DuplicateKeyException` 后，会先寻找是否有该异常的处理方法，若没有则寻找其父类，依次向上寻找，直至到达 `Exception`。

# 数据库中的数据存入 Map

需求: 统计员工性别人数

```java
<select id="countEmpJobData" resultType="java.util.Map">
    select
        (case
            when gender = 1 then '男性员工'
            when gender = 2 then '女性员工'
            end) gender,
        count(0) num
    from emp group by gender;
</select>
```

查询结果:

| gender | num |
| :----: | :-: |
|  男性员工  | 28  |
|  女性员工  |  3  |

```java
// ========= Mapper =========
// 统计员工性别人数
List<Map<String, Object>> countEmpGenderData();

/*
得到的数据格式:
[
	{"gender" = 男性员工, "num" = 28},
	{"gender" = 女性员工, "num" = 3}
]
*/
```

如需将其转换成如下格式

```json
{
	'gender': ['男性员工', '女性员工'],
	'num'= [28, 3]
}
```

可以如下操作:

```java
// 调用 mapper 接口获取统计数据  
List<Map<String, Object>> list = empMapper.countGenderJobData();
// 组装结果并返回  
List<Object> genderList = list.stream().map(dataMap -> dataMap.get("gender")).toList();  
List<Object> numList = list.stream().map(dataMap -> dataMap.get("num")).toList();  
```

最后对其进行封装返回即可。

对 `list` 进行 `stream` 流操作，并用 `map` 映射，将其中每一项元素的指定部分取出，最后使用 `toList` 方法将取出的所有元素封装为 `list`。

# 登录校验

![springboot_登录校验1](img/springboot_登录校验1.png)

## 会话技术

- **会话**: 用户打开浏览器，访问 web 服务器的资源，会话建立，直到有一方断开连接，会话结束。
  在一次会话中可以包含多次请求和响应。
- **会话跟踪**: 一种维护浏览器状态的方法。
  服务器需要识别多次请求是否来自同一浏览器，以便在同一次会话的多次请求间共享数据。
- **会话跟踪方案**:
	- 客户端会话跟踪技术: Cookie
	- 服务端会话跟踪技术: Session
	- 令牌技术

### Cookie

![springboot_登录验证_会话技术_Cookie1](img/springboot_登录校验2.png)

使用流程:
1. 客户端通过浏览器向服务器发起 HTTP 请求。
2. 服务器端在返回浏览器端的 HTTP *响应头* 中添加 `set-Cookie` 字段。
3. 浏览器接收到 `set-Cookie` 字段后将产生 Cookie 并保存在浏览器本地。
4. 当用户再次通过浏览器发送请求后，浏览器验证了 Cookie 中的使用条件后，会在发起的 HTTP 请求的 *请求头* 中添加 `Cookie` 字段及其中的信息。
5. 服务器将接收到的请求进行验证，并返回信息。

**优点**: HTTP协议中支持的技术

**缺点**:
- 移动端APP无法识别Cookie
- 不安全，用户可以自己禁用Cookie
- Cookie不能跨域

> [!tip] 跨域
> 跨域区分三个维度: 协议、IP/域名、端口
> 
> 其中任何一项不同，就代表跨域。

> [!tip] Cookie不能跨域
> 现代网站基本采用 **前后端分离** 技术，前端的代码存储在前端服务器，后端存储在后端服务器，两台服务器的地址不同，端口也不相同。
> 
> 每个 Cookie 都绑定了固定的 协议、域名、端口，而由于前后端分离，两台服务器的 协议、域名、端口 都可能不同，造成了信息的不同步，所以 Cookie 不能跨域。

### Session

![springboot_登录校验3](img/springboot_登录校验3.png)

**优点**: 存储在服务端，安全

**缺点**:
- 服务器集群下无法直接使用Session
- Cookie的缺点

Session 是进阶的 Cookie，也是基于 Cookie 实现的。

但 Session 不再将具体的数据存储在本地浏览器中，而是存在远程服务器端，本地仅存储 `JSESSIONID` 值，在向服务器发起请求时附上该值，服务器端通过该值来进行验证。

### 令牌技术(主流技术)

![新建springboot5](img/springboot_登录校验4.png)

**优点**:
- 支持PC端、移动端
- 解决集群环境下的认证问题
- 减轻服务器端的存储压力

**缺点**: 自己实现

## JWT令牌

全称: JSON Web Token ([http://jwt.io/](http://jwt.io/))

定义了一种简洁的、自包含的格式，用在通信双方以 json 数据格式安全的传输信息。

组成:
1. **Header**: 头，记录 令牌类型、签名算法等。
   例如: `{"alg":"HS256", "type":"JWT"}`
2. **Payload**: 有效载荷，携带一些自定义信息、默认信息等。
   例如: `{"id":"1", "username":"Tom"}`
3. **Signature**: 签名，防止 Token 被篡改、确保安全性。
   将 Header、Payload 融入，并加入指定秘钥，通过指定签名算法计算而来。

![springboot_登录校验5](img/springboot_登录校验5.png)

上图中的字符串由各部分信息通过 `Base64` 算法计算得到。

> [!tip] Base64
> Base64是一种基于64个可打印字符(A-Z a-z 0-9 + /) 来表示二进制数据的编码方式。
> 
> 图中的 = 为补位符号。

### 生成/解析

引入依赖

```xml
<dependency>  
    <groupId>io.jsonwebtoken</groupId>  
    <artifactId>jjwt</artifactId>  
    <version>0.9.1</version>  
</dependency>
```

调用官方提供的工具类 Jwts 来生成或解析 jwt 令牌。

生成 JWT 令牌:

```java
@Test  
public void testGenerateJwt(){  
    Map<String, Object> dataMap = new HashMap<>();  
    dataMap.put("id", 1);  
    dataMap.put("username", "admin");  
    String jwt = Jwts.builder()  
            .signWith(SignatureAlgorithm.HS256, "aXRoZWltYQ==") // 指定加密算法,秘钥  
            .addClaims(dataMap) // 添加自定义信息  
            .setExpiration(new Date(System.currentTimeMillis() + 3600 * 1000)) // 设置过期时间  
            .compact(); // 生成令牌  
    System.out.println(jwt);  
  
/*
eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsImV4cCI6MTc5Mzc5MzIwNX0.eysulL-atple6rN4wdbXYvSWJTB03IQap4BFQQlFHGQ  
*/
}
```

解析 JWT 令牌: 

```java
@Test  
public void testParseJWT(){  
    String token = "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MSwidXNlcm5hbWUiOiJhZG1pbiIsImV4cCI6MTc5Mzc5MzIwNX0.eysulL-atple6rN4wdbXYvSWJTB03IQap4BFQQlFHGQ";  
    Claims claims = Jwts.parser()  
            .setSigningKey("aXRoZWltYQ==") // 指定秘钥  
            .parseClaimsJws(token) // 解析令牌  
            .getBody(); // 获取自定义信息  
    System.out.println(claims);  
}
```

> [!tip]
> 在上面代码中使用到的秘钥 `aXRoZWltYQ==` 是 `itheima` 进行 base64 算法加密后的一串字符。

### 应用

JWT 令牌工具类:

```java
public class JwtUtils {

    /** 签名秘钥 */
    private static final String SIGN_KEY = "aXRoZWltYQ==";

    /** 令牌有效期：12 小时（毫秒） */
    private static final long EXPIRE_TIME = 12 * 60 * 60 * 1000L;

    /** 工具类禁止实例化 */
    private JwtUtils() {
    }

    /**
     * 生成 JWT 令牌
     *
     * @param claims 自定义载荷，例如 {"id": 1, "username": "admin"}
     * @return 生成的 token 字符串
     */
    public static String generateToken(Map<String, Object> claims) {
        return Jwts.builder()
                .signWith(SignatureAlgorithm.HS256, SIGN_KEY) // 指定加密算法、秘钥
                .addClaims(claims)                             // 添加自定义信息
                .setExpiration(new Date(System.currentTimeMillis() + EXPIRE_TIME)) // 12 小时后过期
                .compact();
    }

    /**
     * 解析 JWT 令牌
     *
     * @param token 令牌字符串
     * @return 载荷 Claims，可通过 claims.get("id") 等方式取值
     */
    public static Claims parseToken(String token) {
        return Jwts.parser()
                .setSigningKey(SIGN_KEY)   // 指定秘钥
                .parseClaimsJws(token)     // 解析令牌
                .getBody();                // 获取载荷
    }
}
```

三层架构:

```java
// ============== controller ==============
@Slf4j  
@RestController  
@RequestMapping("/login")  
public class LoginController {  
  
    @Autowired  
    private EmpService empService;  
  
    @PostMapping  
    public Result login(@RequestBody Emp emp){  
        log.info("登录: {}", emp);  
        LoginInfo info = empService.login(emp);  
        if(info != null){  
            return Result.success(info);  
        }  
        return Result.error("用户名或密码错误");  
    }  
}


// ============== service ==============
@Override  
public LoginInfo login(Emp emp) {  
    // 根据用户名和密码查询员工信息  
    Emp e = empMapper.selectByUsernameAndPassword(emp);  
    // 判断是否存在这个员工  
    // - 存在，组装员工信息    
    if(e != null){  
        log.info("登录成功，员工信息: {}", e);  
  
        // 生成 JWT 令牌  
        Map<String, Object> claims = new HashMap<>();  
        claims.put("id", e.getId());  
        claims.put("username", e.getUsername());  
        String jwt = JwtUtils.generateToken(claims);  
  
        return new LoginInfo(e.getId(), e.getUsername(), e.getName(), jwt);  
    }  
  
    // - 不存在，返回 null  
    log.info("登录失败，用户名或密码错误");  
    return null;  
}


// ============== mapper ==============
/**  
 * 根据用户名和密码查询员工信息 * @param emp  
 * @return  
 */  
Emp selectByUsernameAndPassword(Emp emp);

// XML 
<select id="selectByUsernameAndPassword" resultType="com.itheima.pojo.Emp">  
    select id, username, name from emp where username = #{username} and password = #{password}  
</select>
```

## 过滤器 Filter

Filter过滤器，是 JavaWeb 三大组件(Servlet、Filter、Listener)之一。

![springboot_登录校验6|816](img/springboot_登录校验6.svg)

过滤器可以把对资源的请求拦截下来、从而实现一些特殊的请求。

过滤器一般完成一些 *通用* 的操作，比如: 登录校验、统一编码处理、敏感字符处理等。

### 快速入门

定义 Filter: 定义一个类，实现 Filter 接口，并实现其所有方法。

配置 Filter: 
- Filter 类上加 `@WebFilter` 注解，配置拦截路径。
- 引导类上加上 `@ServletComponentScan` 开启 Servlet 组件支持。

```java
package com.itheima.filter;

import jakarta.servlet.*;
import jakarta.servlet.annotation.WebFilter;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

@Slf4j
@WebFilter(urlPatterns = "/*") // 拦截所有请求
public class DemoFilter implements Filter { // Filter(of jakarta.servlet)
    // 初始化方法，web服务器启动时执行，只执行一次
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        log.info("init 初始化方法...");
    }

    // 拦截到请求之后执行，可执行多次
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        log.info("拦截到了请求...");
        // 放行
        chain.doFilter(request, response);
    }

    // 销毁方法，web服务器关闭时执行，只执行一次
    @Override
    public void destroy() {
        log.info("destroy 销毁方法...");
    }
}
```

```java
@ServletComponentScan // 开启springboot对servlet组件的支持
@SpringBootApplication
public class TliasWebManagementApplication {

    public static void main(String[] args) {
        SpringApplication.run(TliasWebManagementApplication.class, args);
    }
}
```

### 实现令牌校验

工作流程:

![springboot_登录校验7|611](img/springboot_登录校验7.svg)

代码:

```java
package com.itheima.filter;

import com.itheima.utils.JwtUtils;
import jakarta.servlet.*;
import jakarta.servlet.annotation.WebFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

@Slf4j
@WebFilter(urlPatterns = "/*")
public class TokenFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        HttpServletRequest req = (HttpServletRequest) request;
        HttpServletResponse resp = (HttpServletResponse) response;

        // 1. 获取到请求路径
        String requestURI = req.getRequestURI(); // /login

        // 2. 判断是否是登录请求，如果路径中包含 /login，说明是登录操作，放行
        if(requestURI.contains("/login")){
            log.info("登录请求，放行");
            chain.doFilter(request, response);
            return;
        }

        // 3. 获取请求头中的 token
        String token = req.getHeader("token");

        // 4. 判断 token 是否存在，如果不存在，说明用户没有登录，返回错误信息(响应401状态码)
        if(token == null || token.isEmpty()){
            log.info("令牌为空，响应 401 状态吗");
            resp.setStatus(401);
            return;
        }

        // 5. 如果 token 存在，校验令牌，如果校验失败 -> 返回错误信息(响应 401 状态码)
        try {
            JwtUtils.parseToken(token);
        }catch (Exception e){
            log.info("令牌非法，响应 401 状态吗");
            resp.setStatus(401);
            return;
        }

        // 6. 校验通过，放行
        log.info("令牌合法，放行");
        chain.doFilter(request, response);
    }
}
```

### 执行流程

```java
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        log.info("拦截到了请求...");
        System.out.println("执行放行前的逻辑...");
        // 放行
        chain.doFilter(request, response);

        System.out.println("执行放行后的逻辑...");
    }
```

![springboot_登录校验8](img/springboot_登录校验8.svg)

1. Filter 首先拦截浏览器的请求，执行放行前的逻辑，如判断是否登录等。
2. Filter 放行浏览器的请求，访问服务器中的资源。
3. 回到 Filter 中，继续执行 Filter 放行后的逻辑。

### 拦截路径

```java
@WebFilter(urlPatterns = "/*")
public class DemoFilter implements Filter {
```

常见的拦截资源路径:

|  拦截路径  | urlPatterns值 |          含义           |
| :----: | :----------: | :-------------------: |
| 拦截具体路径 |    /login    | 只有访问 /login 路径时，才会拦截  |
|  目录拦截  |   /emps/*    | 访问 /emps 下的所有资源，都会被拦截 |
|  拦截所有  |      /*      |     访问所有资源，都会被拦截      |

### 过滤器链

一个 web 应用中，可以配置多个过滤器，这多个过滤器就形成了 **过滤器链**。

![springboot_登录校验9](img/springboot_登录校验9.svg)

顺序: 注解配置的 Filter，优先级是按照过滤器类名(字符串)的自然排序。

例如有 `AbcFilter`、`DemoFilter` 这两个 Filter，同时拦截的都是 `/*`，那么就会先执行 `AbcFilter` 后执行 `DemoFilter`。(将 字符串 "AbcFilter" 和 "DemoFilter" 字典序排序后，"AbcFilter" 更靠前)

## 拦截器 Interceptor

Interceptor 是一种动态拦截方法调用的机制，类似于过滤器。

Spring 框架中提供的，主要用来动态拦截控制器方法的执行。

![springboot_登录校验10](img/springboot_登录校验10.svg)

作用: 拦截请求，在指定方法调用前后，根据业务需要执行预先设定的代码。

### 快速入门

1.  定义拦截器，实现 `HandlerInterceptor` 接口，并实现其所有方法。

```java
@Slf4j  
@Component  
public class DemoInterceptor implements HandlerInterceptor {  
    // 在目标资源方法(Controller)之前运行 - false 表示拦截，true 表示放行  
    @Override  
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {  
        log.info("preHandle...");  
        return true;  
    }  
  
    // 在目标资源方案运行之后运行  
    @Override  
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, @Nullable ModelAndView modelAndView) throws Exception {  
        log.info("postHandle...");  
    }  
  
    // 视图渲染完毕后运行，使用于 非前后端分离架构
    @Override  
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, @Nullable Exception ex) throws Exception {  
        log.info("afterCompletion...");  
    }  
}
```

2. 注册拦截器。

```java
/**  
 * 配置类 
*/
@Configuration  // 继承了 @Component
public class WebConfig implements WebMvcConfigurer {  
  
    @Autowired  
    private DemoInterceptor demoInterceptor;  
  
    @Override  
    public void addInterceptors(InterceptorRegistry registry) {  
        registry.addInterceptor(demoInterceptor)  // 增加拦截器
                .addPathPatterns("/**"); // 拦截所有请求  
    }  
}
```

![springboot_登录校验11](img/springboot_登录校验11.png)

### 实现令牌校验

工作流程:

![springboot_登录校验7](img/springboot_登录校验7.svg)

代码: 

```java
/**
 * 令牌校验的拦截器
 */
@Slf4j
@Component
public class TokenInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        // 1. 获取到请求路径
        String requestURI = request.getRequestURI(); // /login

        // 2. 判断是否是登录请求，如果路径中包含 /login，说明是登录操作，放行
        if(requestURI.contains("/login")){
            log.info("登录请求，放行");
            return true;
        }

        // 3. 获取请求头中的 token
        String token = request.getHeader("token");

        // 4. 判断 token 是否存在，如果不存在，说明用户没有登录，返回错误信息(响应401状态码)
        if(token == null || token.isEmpty()){
            log.info("令牌为空，响应 401 状态吗");
            response.setStatus(401);
            return false;
        }

        // 5. 如果 token 存在，校验令牌，如果校验失败 -> 返回错误信息(响应 401 状态码)
        try {
            JwtUtils.parseToken(token);
        }catch (Exception e){
            log.info("令牌非法，响应 401 状态吗");
            response.setStatus(401);
            return false;
        }

        // 6. 校验通过，放行
        log.info("令牌合法，放行");
        return true;
    }
}
```

### 拦截路径

拦截器可以根据需求，配置不同的拦截路径。

|    拦截路径     |        含义        |                         举例                          |
| :---------: | :--------------: | :-------------------------------------------------: |
|    `/*`     |       一级路径       |    能匹配 `/depts` `/emps` `/login`，不能匹配 `/dept/1`     |
|    `/**`    |      任意级路径       |        能匹配 `/depts` `/depts/1` `/depts/1/2`         |
| `/depts/*`  | `/depts` 下的一级路径  |      能匹配 `/depts/1`，不能匹配 `/depts/1/2` `/depts`      |
| `/depts/**` | `/depts` 下的任意级路径 | 能匹配 `/depts` `/depts/1` `/depts/1/2`，不能匹配 `/emps/1` |

通过 `excludePathPatterns()` 方法，可以设置哪些请求不拦截。

```java
/**
 * 配置类
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Autowired
    private TokenInterceptor tokenInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(tokenInterceptor)
                .addPathPatterns("/**") // 拦截所有请求
                .excludePathPatterns("/login"); // 不拦截哪些请求
    }
}
```

那么在拦截器的实现代码中的前两步就可以省略。

```java
/**
 * 令牌校验的拦截器
 */
@Slf4j
@Component
public class TokenInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        // // 1. 获取到请求路径
        // String requestURI = request.getRequestURI(); // /login
        //
        // // 2. 判断是否是登录请求，如果路径中包含 /login，说明是登录操作，放行
        // if(requestURI.contains("/login")){
        //     log.info("登录请求，放行");
        //     return true;
        // }

        // 3. 获取请求头中的 token
        String token = request.getHeader("token");

        // 4. 判断 token 是否存在，如果不存在，说明用户没有登录，返回错误信息(响应401状态码)
        if(token == null || token.isEmpty()){
            log.info("令牌为空，响应 401 状态吗");
            response.setStatus(401);
            return false;
        }

        // 5. 如果 token 存在，校验令牌，如果校验失败 -> 返回错误信息(响应 401 状态码)
        try {
            JwtUtils.parseToken(token);
        }catch (Exception e){
            log.info("令牌非法，响应 401 状态吗");
            response.setStatus(401);
            return false;
        }

        // 6. 校验通过，放行
        log.info("令牌合法，放行");
        return true;
    }
}
```

### 执行流程

![springboot_登录校验12](img/springboot_登录校验12.svg)

## Filter 与 Interceptor 区别

1. 接口规范不同: 过滤器需要实现 Filter 接口，而拦截器需要实现 `HandlerInterceptor` 接口。
2. 拦截范围不同: 过滤器 Filter 会拦截所有的资源，而 `Interceptor` 只会拦截 Spring 环境中的资源。