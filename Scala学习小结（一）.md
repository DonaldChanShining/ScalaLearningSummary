# 一、	前言
Scala 是一种混合了面向对象编程和函数式编程的特性的语言，由于我本人是从java转向Scala的，所以本文侧重于介绍总结和展示我在学习过程中使用函数式编程的一些体验。  

首先,函数式编程和面向对象编程只是软件开发的两种不同途径，并不对立。面向对象编程可以认为是一种自顶向下的程序设计方法，
我们一般通过名词（抽象为使用的对象）做切割，每个对象有状态，行为，标识符等，定义名词后，再进一步定义名词之间的交互，我们
需要将交互也放入对象中，通常会定义为服务类进行操作，一切皆为对象。而函数式编程中，一般将软件分解为需要执行的行为和操作，
可以认为是一种自底向上的设计方法，函数可以看作程序中的基本元素，取代了我们通常理解的变量。函数纯粹对输入进行操作，产生结
果，所有变量视为不可变的，并且尽可能的将副作用（状态的变化）推迟或消除，从而使得容易对程序进行推理。按照这个思路，我们可
以发现，函数式编程中单元测试和debug的难度大大降低了，因为变量不可变，所以程序运行中不会修改任何别的东西，决定函数结果的只
有输入，因而调试中的错误是百分之百不用花很多精力可以实现重现的，同时每个函数都可以单独的通过单元测试。还有一个重要的好处是
可以实现热部署，FP中所有状态都是传给函数的参数，而参数都是存储在栈上的，所以理论上只需要比较正在运行的代码和更新的代码获得
一个diff，然后利用这个diff更新现有代码，其他的交给语言工具自动完成就行了！  

下面我会从高阶函数，模式匹配，集合，数据结构，综合使用，框架的使用demo这六个我认为比较有代表性的方面来写这篇文章。  

# 二、 高阶函数
在数学和计算机科学中，高阶函数是至少满足下列一个条件的函数：a.接受一个或多个函数作为输入,b.输出一个函数。在数学中它们也叫做算子（运算符）或泛函。微积分中的导数就是常见的例子，因为它映射一个函数到另一个函数。在无类型lambda演算，所有函数都是高阶的；在有类型lambda演算（大多数函数式编程语言都从中演化而来）中，高阶函数一般是那些函数型别包含多于一个箭头的函数。在函数式编程中，返回另一个函数的高阶函数被称为Curry化的函数。在很多函数式编程语言中能找到的map函数是高阶函数的一个例子。它接受一个函数f作为参数，并返回接受一个列表并应用f到它的每个元素的一个函数。  

通俗的讲，我们可以将高阶函数理解为可以接受函数作为参数传入，也可以将函数作为结果输出的函数，即函数为一等公民，函数可以在任何地方定义，在函数内或函数外，可以作为函数的参数和返回值，可以对函数进行任意组合。  

可能有些人和我一样有这样的疑问：道理我都懂，可是这些特性真的有用吗？事实上，一切皆视为函数可以进一步增加我们的代码的可重用性，将重用的细粒度提升到了函数，与此同时，在数据的处理方面拥有着巨大的优势。下面通过几段对比代码来显示高阶函数和函数式编程的有趣之处。  

我们来到一个兑换零钱的需求场景，首先你有一个金币总和，请问有多少种方法可以兑换。按照常识，金币的面值有1,2,5.于是乎，我们写出了下面的代码。
``` java
public static int countChange(int sum)  {
    int s=0;
    for(int i=0;i<=sum/5;i++){
        for(int j=0;j<=(sum-i*5)/2;j++){
            for(int k=0;k<=(sum-i*5-j*2);k++){
                if(5*i+2*j+k==sum){
                    s++;
                    System.out.println(i+"*5 + "+j+"*2 + "+k+"*1 = "+sum);
                }
            }
        }
    }
    return s;
}
```
看起来很美，但是过了几天领导告诉你，需求变了，要添加别的面值的金币。没关系，我可以继续嵌套for循环进去啊，终于有一天，代码臃肿到你不能忍了，于是你想到，我要去写一个通用的方法。于是乎，你对方法进行了重构。  

这个时候，我们换个思路，来到了函数式编程中，便可以写出如下的代码:
``` scala
def countChange(money: Int, coins: List[Int]): Int = {
  val l = coins.sorted.reverse
  if (l.isEmpty || money < 0) 0
  else if (l.tail.isEmpty && money % l.head == 0) 1
  else countChange(money - l.head, l) + countChange(money, l.tail)
}
```
是不是简洁了许多，只需要一行代码，就能完成对集合的排序操作，然后，你只需要理顺逻辑需求，告诉机器需要做什么就可以了，至于如何做剩下的就交给计算机来计算了。其实，这就是编程中的一种递归的思想，当然我们还可以继续优化，此处就不做继续赘述了。  

我们来到下一个经典的例子，斐波拉契数列（0,1,1,2,3，…），我们可以按照定义用递归的方式将数列进行如下描述：  
``` scala
def fib(n: Int): Long = {
  if (n == 1) 0
  else if (n == 2) 1
  else fib(n - 1) + fib(n - 2)
}
```
我们也可以使用用递归优化过后的形式--尾递归来表示：
``` scala
def fibb(n: Int): Long = {
	def fib(n: Int, res1: Long, res2: Long): Long = {
      if (n == 1) res1
      else ff(n - 1, res2, res1 + res2)
    }
  fib(n, 0, 1)
}
```
可能就代码而言可能体现不出尾递归的优势，我们同时使用两个函数，对第50个斐波拉契数进行计算，计算和耗时结果如下：
``` scala
    t1 : Long = 7778742049
    time1 : Long = 50935
    t2 : Long = 7778742049
    time2 L Long = 2
```
t1，time1和t2,time2分别表示fib（50）和fibb(50)的计算结果和耗时,我们可以很直观的看出由于效率导致的计算时间上的差距，
下面我们简单分析下为何会出现如此大的差距，在fib方法中， fib方法会不断地调用自身，直至满足条件到递归出口，但是在这个过程中，
机器会不断地开辟空间保存上一次调用时的位置，这样才能最后一层层的返回结果给函数，所以说如果递归调用的层数特别深，就可能出现
抛出栈溢出异常的情况，而我们再看尾递归，每次的返回结果都包含了当前的计算情况，这样每次递归调用就相当于更新一个堆栈上面的状态，
不需要保留上一次的具体情景，从而大大减小了内存上的开销。而函数式编程的一个优势就在于可以进行尾递归调优，既可以将繁复的迭代转化为
递归，又可以不损失效率。
 
高阶函数中，我们必须要提要的两个重要的概念，一个就是--闭包（closure），函数定义和表达式位于另一个函数的函数体内，而且，这些内部函数
可以访问他们所在的外部函数的所有局部变量，和声明的其他内部函数，当这样的内部函数在包含它们的外部函数之外被调用时，就形成了闭包。上文中使用尾递归的斐波拉契数列函数即可看作闭包的。

还有一个不得不提的就是---柯里化（currying），将原来接受多个参数的函数变成新的接受一个参数的函数的过程，
新的函数返回一个以原有的第二个参数为参数的函数。可能有点绕口，我们可以用一个表达式直观的表示：
``` scala
def f(args1)(args2)...(argsn) = E
==>     def f = (args1=>(args2=>...(argsn => E)...))
```
由于文章开始时提到过，函数在函数式编程中是一等公民，所以可以理解拥有这些特性。但是这时候又会产生一个疑问，这个闭包和柯里化的优点是什么呢，下面我们结合例子说明，分别定义求平方和立方的函数：
``` scala
def product(x:Int) : Int = {
    x*x
}
def cube(x:Int) : Int = {
    x*x*x
}
```
如果我们需要定义很多这种模式的函数，我们是否可以应用函数式编程的特性构造一些类似于基类的基函数，于是我们构造了下面这样一个函数：
``` scala
def base(a:Int,b:Int):Int={
    def loop(sum:Int,b:Int):Int = {
        if(b==0) 1
        else if(b==1) sum
        else loop(a*sum,b-1)
    }
    loop(a,b)
}
```
我们就可以利用柯里化这样定义平方和立方的函数了：
``` scala
def product(x:Int) = base(x，2）
def cube(x:Int) = base(x,3)
```
哈，这样你一定感受到了什么吧，对，函数式编程的好处之一就是提升了代码复用的细粒度，将可复用的代码从方法提升到了函数，
而正是因为柯里化和闭包这些函数式语言特性的支持才能做到。下面我们举几个好玩和实用的函数给大家：
求n的阶乘
``` scala
(1 to n).reduceLeft(_ * _)
```
打印*组成的三角形
``` scala
(1 to 9).map("*" * _).foreach(println _)
```
用50个字符内的代码求100内的奇数和
```scala
(1 to 100).filter(_ % 2 != 0).map(x => x*x).sum
```

# 三、 模式匹配 
模式匹配现在渐渐成了函数式编程的一个标志性特性，为了说明这个特性，我们首先回顾一下java中的switch语句，在java编程时，有个常用的技巧就是每个方法只有一个返回点，但这意味着如果方法里有某种条件逻辑，我们就需要创建一个变量来存放最终的返回值，方法执行时，变量更新为方法要返回的值，在方法的最后return变量。反映到代码中就是：
```java
public static String createMsg(int errorCode){
    String res = "";
    switch (errorCode){
        case 1: res = "yeah";
        case 2: res = "hello";
        case 3: res = "noob";
        default: res = "what's up";
    }
    return res;
}
```
我们再看一下使用模式匹配后的代码：
```scala
def createMsg(errorCode: Int) = errorCode match {
  case 1 =>  "yeah"
  case 2 =>  "hello"
  case 3 =>  "noob"
  case _ =>  "what's up"
}
```
可以对比发现，代码简洁了许多，也许有人疑问，没有返回类型真的没问题吗，实际上模式匹配上返回一个值，类型会为所有case语句分支返回的值得公共超类，如果有一个模式没有匹配上的话，就会在编译时抛出异常。`所以说模式匹配某种意义上可以看做一种更高级的switch，而且适用范围远大于传统意义的switch，例如java中只能接收数值或者枚举类型，而在scala中却并没有这些限制，配合上样例类的使用，可以大大简化我们的开发流程。例如我们下面举的根据将json进行转换打印的例子：
```scala
abstract class JSON{
  def show:String = this match {
    case JSeq(elems)=> "[" + (elems map (_.show) mkString ",") + "]"
    case JObj(bindings) =>
      val assocs = bindings map {
        case (key,value) => "\"" + key +"\": " + value.show
      }
      "{" + (assocs mkString ", ") + "}"
    case JNum(num) => num.toString
    case JStr(str) => "\"" + str + "\""
    case JBool(b) => b.toString
    case JNull => "null"
  }
}
case class JSeq(elems:List[JSON]) extends JSON
case class JObj(bindings:Map[String,JSON]) extends JSON
case class JNum(num:Int) extends JSON
case class JStr(str:String) extends JSON
case class JBool(b:Boolean) extends JSON
case object JNull extends JSON
```
下面是测试代码,定义一个json，然后转换后打印出来:
```scala
val data = JObj(Map(
  "firstName" -> JStr("Yu"),
  "lastName" -> JStr("Gong"),
  "address" -> JObj(Map(
    "streetAddress" -> JStr("NY"),
    "state" -> JStr("NY")
  )),
  "phoneNumbers" -> JSeq(List(
    JObj(Map(
      "type" -> JStr("home"), "number" -> JStr("12233")
    )),
    JObj(Map(
      "type" -> JStr("fax"), "number" -> JStr("22222")
    ))
  ))
))
println(data.show)

{"firstName": "Yu", "lastName": "Gong", "address": {"streetAddress": "NY", "state": "NY"}, "phoneNumbers": [{"type": "home", "number": "12233"},{"type": "fax", "number": "22222"}]}
```
试想一下如果我们用java需要怎样的工作量，我们也可以类似的定义基类JSON及其子类，如果实现show函数，则需要在每个子类中实现自己的打印函数，代码量和代码复杂度大大增加了。同时，这里可以体现出样例类组合使用的几点好处：a.模式匹配通常比继承更容易把我们引向精简的代码;b.构造时不需要new的复合对象更加易读;c.我们将直接获得toString,equals,hashCode和copy方法，并且可以工作的和我们期望的一样（不用重写hashcode和equals方法了~）。当我们需要处理一大堆程序分支时或者处理需要嵌套一大堆if-else语句时，再也不用和括号海和臃肿的挤在一坨的代码做斗争了，模式匹配可以更好的更清晰完成任务，当需要修改业务逻辑时，例如添加分支或改分支，也不用回顾以前的冗余的代码了，插入的地方一目了然,大大减轻了开发和维护的代价。

#四、 集合的操作
在一本书上看到如此称赞scala的集合库，“Scala集合库是Scala生态系统里最棒的库，没有之一”，我个人也是深以为然，由于高阶函数的存在，scala的开发者在它的集合中提供了无数有用的函数，以及大量不同的存储和操作数据的方法，使得一行代码可以完成原来不知道多少行才能完成的事情，同时效率也很高。

一般的对集合的处理上，如果是java一般都是采用for循环或者Iterator获取一个迭代器，然后进行操作，在迭代器使用方面，scala和java没有明显的区别， 而在For的使用方面，却有着本质上的区别，java中for即for循环（for each循环，，，回去看编程思想看本质。。）  scala中由于有高阶函数的支持，for只是看做一种语法糖。每一句生成一个generator来遍历对应的序列。

#五、 几个重要的数据结构
下面我将列举集合在Scala中经常用到的数据类型，Option和Either,List和Vector
##Option
Option是scala标准库中的一个容器，通过使用Option来鼓励大家一般编程时尽量不会因为Null产生不必要的错误，例如烦人的空指针异常。Option通过两个子类来实现这个含义，Some和None.Some表示容器有且仅有一个东西，None表示空容器。当取出Option中的内容时，我们可以使用get和getOrElse，getOrElse（）可以指定如果没取到时的返回的默认值，这样就可以避免不必要的错误了。同时Option的伴生类里面提供了工厂方法，能将java风格的引用转换为Option类型。源码定义如下：
```scala
def apply[A](x: A): Option[A] = if (x == null) None else Some(x)
```
这个时候你可能有疑问，我们循环时用if习惯性的检查一下Null不就可以了吗，实际上，scala设计时在Option中提供了一系列的高级特性，使得简单很多。Option最重要的特性就是可以被当做集合来看待，意味着，我们可以使用上一节介绍过的map,.flatMap,foreach等方法，也可以将这个结构用在 for推倒式里面，即优雅简洁，又安全。  

下面引用一个depth in scala中的情景:我们写代码时有很多地方需要在某变量有值时构建某结果，无值时构建另外一个默认值，所以我们来到这样一个场景，有个应用在执行时需要某种临时文件存储，应用设计为用户能在命令行形式下提供可选的参数。
```scala
def getTemporaryDirectory(tmpArg:Option[String]):java.io.File = {
  tmpArg.map{ name => new java.io.File(name)}.
    filter(_.isDirectory).
    getOrElse(new File(System.getProperty("java.io.tmpdir")))
}
```
只需要4行，我们就完成了一个比较健壮的方法（当然，也可以认为是一行，因为美观折叠了下）。如果用java,我们需要各种判断和嵌套判断是否为空，使得程序的简洁性大打折扣，写起来也是很繁琐，如果有遗漏判断，又需要在里面包一层if。

##Either
我们在程序的编写中，如果把每个错误都当异常处理会使代码变得凌乱，如果认为有返回值就是正常运行返回的标志也不合适，于是Scala中引入了不相交并集来从方法中返回结果，Either就是其中一种。不相交并集将两个完全不同的类型结合在一起，一种类型表示成功并携带返回值，另一种类型表示失败并持有失败信息，在调用某种方法时，我们可以得到这两个并集之一，展开便知道调用具体情况。类似于Option,Either的定义如下：
```scala
sealed trait Either[+E,+A]
case class Left[+E](value:E) extends Either[E，Nothing]
case class Right[+A](value:A) extends Either[Nothing,A]
```
Either将Left和Right类型组合在一起，由于创建者的原因，一般用Left表示错误，Right表示正确。我们用一个例子来描述Either的使用方法：
```scala
def division(i:Int) = {
    if(i==0）Left("divisor can`t be 0")
    else Right(20/i)
}
def test(n:Int) = {
    division(n) match {
        case Left(reason) => s"Failed: $reason"
        case Right(result) => result
    }
}
```
可以发现，如果正确返回，可以同Option类似的从Right中取出值，如果不对，可以从Left中取出错误原因，这样的设计和Option有着异曲同工之妙，从某种程度上讲这种设计模式可以认为是优雅的处理错误，将Left和模式匹配结合起来，可以使异常的处理代码不再那么不堪，同时，结合map,flatmap,withFilter的使用大大的增加了处理问题的灵活性。Scala中还有一些这样类似设计的数据结构，我这里就不做太多赘述了，像Try，更是将两个子类直接定义成了Success和Failure来类似的处理正确和错误的信息。

##List

##Vector

#六、 综合应用
上面絮絮叨叨的描述了一些我个人应用中感受到的函数式编程的优点，下面我用fp in scala中的最后一个例子来综合的对比下scala和java。  

取水问题：我们有一个无限滴水的水龙头，有若干个已知容量的水杯，如何操作取出一定量的水？（Eg:一个水杯4L，一个水杯9L，我们如何操作取出6L水）对于这个问题，我们可以抽象出每一步水杯之间交互倒水的动作，然后使用枚举法列出每一步操作可能出现的结果，发散出去，当有一个杯子的水等于目标容量时终止!这种做法不一定是最好的，但是我们用相同的思路分别用Scala 和java进行一遍操作。   
下面是scala的相关实践和测试代码。
```scala
class Pouring(capacity:Vector[Int]){
  trait Move {
    def change(state:State):State
  }
  case class Empty(glass:Int) extends Move{
    def change(state:State) : State = state updated (glass,0)
  }
  case class Full(glass:Int) extends Move{
    def change(state:State) : State = state updated (glass,capacity(glass))
  }
  case class Pour(from:Int,to:Int) extends Move{
    def change(state:State) : State = {
      val amount = state(from) min (capacity(to) - state(to))
      state.updated(from,state(from) - amount).
        updated (to,state(to) + amount)
    }
  }
  type State = Vector[Int]
  val initialState = capacity map (x => 0)
  val glasses = 0 until capacity.length
  val moves =
    (for (g<- glasses) yield Empty(g)) ++
      (for (g<- glasses) yield Full(g)) ++
      (for (from <- glasses;to<-glasses if from != to) yield Pour(from,to))

  class Path(history:List[Move]){
    def endState : State = (history foldRight initialState)(_ change _)
    def extend(move: Move) = new Path(move :: history)
    override def toString = (history.reverse mkString " ") + "-->" + endState
  }

  val initialPath = new Path(Nil)

  def from(paths:Set[Path]):Stream[Set[Path]] = {
    if(paths.isEmpty) Stream.empty
    else{
      val more = for{
        path <- paths
        next <- moves map path.extend
      } yield next
      paths #:: from(more)
    }
  }

  val pathSets = from(Set(initialPath))

  def solution(target:Int):Stream[Path] ={
    for{
      pathSet <- pathSets
      path <- pathSet
      if path.endState contains target
    } yield path
  }
}

object Test extends App{
  val t = System.currentTimeMillis()
  val problem = new Pouring(Vector(4,9))
  val  s =  problem.solution(6)
  println(s.toString() + "，耗时：" + (System.currentTimeMillis()-t))
}
```
下面是java版本的代码和测试程序：
```java
public class PouringWater {

    public static Vector<Integer> CAPACICY;

    public Vector<Integer> initialState;

    public PouringWater(Vector<Integer> ca) {
        CAPACICY = (Vector<Integer>) ca.clone();
        initialState = initial(ca);
    }

    private Vector<Integer> initial(Vector<Integer> ca) {
        int size = ca.size();
        for (int i = 0; i < size; i++) {
            ca.set(i, 0);
        }
        return ca;
    }

    private Vector<Integer> full(int glass,Vector<Integer> oriState) {
        Vector<Integer> state = (Vector<Integer>) oriState.clone();
        state.set(glass,CAPACICY.get(glass));
        return state;
    }

    private Vector<Integer> empty(int glass,Vector<Integer> oriState) {
        Vector<Integer> state = (Vector<Integer>) oriState.clone();
        state.set(glass,0);
        return state;
    }

    private Vector<Integer> pour(int from,int to,Vector<Integer> oriState) {
        Vector<Integer> state = (Vector<Integer>) oriState.clone();
        int amount = Math.min(state.get(from),CAPACICY.get(to)-state.get(to));
        state.set(from,state.get(from)-amount);
        state.set(to,state.get(to)+amount);
        return state;
    }

    public Set<Vector<Integer>> move(Vector<Integer> state){
        int size = state.size();
        Set<Vector<Integer>> conditions = new HashSet<>();
        for(int i = 0; i< size; i++){
            conditions.add(full(i,state));
            conditions.add(empty(i,state));
            for(int j=0;j<size;j++){
                if(i != j) {
                    conditions.add(pour(i, j, state));
                }
            }
        }
        conditions.remove(state);
        return conditions;
    }

    public Map<StringBuilder, Vector<Integer>> oneOperation(Vector<Integer> condition, StringBuilder sb) {
        Map<StringBuilder, Vector<Integer>> maps = new HashMap<>();
        Set<Vector<Integer>> states = move(condition);
        for (Vector<Integer> state : states) {
            StringBuilder oneStepStr = new StringBuilder().append(sb.toString());
            oneStepStr.append("-->").append(state.toString());
            maps.put(oneStepStr, state);
        }
        return maps;
    }

    public StringBuilder solute(int goal, Map<StringBuilder, Vector<Integer>> maps) {
        Map<StringBuilder, Vector<Integer>> oneFloor = new HashMap<>();
        for (StringBuilder biu : maps.keySet()) {
            if (biu.toString().contains(String.valueOf(goal))) {
                return biu;
            }
        }
        for (StringBuilder biubiu : maps.keySet()) {
            oneFloor.putAll(oneOperation(maps.get(biubiu), biubiu));
        }
        return solute(goal, oneFloor);
    }

    public String getSolution(int goal) {
        StringBuilder sb = new StringBuilder().append(initialState);
        Map<StringBuilder, Vector<Integer>> initialMap = new HashMap<>();
        initialMap.put(sb, initialState);
        return solute(goal, initialMap).toString();
    }

    public static void main(String[] args) {
        Vector<Integer> state = new Vector<>();
        state.add(0, 4);
        state.add(1, 9);
        PouringWater p = new PouringWater(state);
        System.out.println(p.getSolution(6));
    }
}
```


#七、 后记
由于笔者也才接触函数式编程一个多月，所以上述描述中可能有不太准确的地方，欢迎大家指正和交流。同时本文只是粗略的总结了一遍scala中笔者目前学习的情况，同时由于为了对比写的一些拙劣的java代码，也希望大家能够给予我优化的方案和讨论（大神们轻拍..）。   

scala里面还有一些高级的特性，如moniods,monad，Continuation，I/O的处理，stream的深入使用，由于笔者还未完全掌握，所以在日后在有一定的理解后会更新相关方面，框架中akka，spray,slick，play等框架的使用demo，函数式编程的一些思考方式亦会在后续学习文章中更新。 

               











