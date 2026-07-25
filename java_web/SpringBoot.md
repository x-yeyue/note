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

