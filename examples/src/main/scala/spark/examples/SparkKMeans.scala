package spark.examples

import java.util.Random
import scala.math.exp
import Vector._
import spark._
import spark.broadcast._

object SparkKMeans {
  val N = 10000  // Number of data points
  val D = 10   // Numer of dimensions
  val R = 0.7  // Scaling factor
  val K = 4   // Num of Clusters
  val rand = new Random(42)

  case class DataPoint(x: Vector, y: Double){
    def mag():Double = {
      return math.sqrt(x.dot(x) + y*y)
    }
    def - (other: DataPoint): DataPoint = {
      return DataPoint(this.x-other.x, this.y-other.y)
    }
    def + (other: DataPoint): DataPoint = {
      return DataPoint(this.x+other.x, this.y+other.y)
    }
    def * (other: Double): DataPoint = {
      return DataPoint(this.x*other, this.y*other)
    }
  }
  object DataPoint{
    implicit object DataPointAccumParam extends spark.AccumulatorParam[DataPoint] {
      def addInPlace(t1: DataPoint, t2: DataPoint) = t1 + t2
      def zero(initialValue: DataPoint) = DataPoint(Vector.zeros(initialValue.x.length), 0)
    }
  }

  def randPoint(dev:Double, center: Double): DataPoint = {
    val y = rand.nextGaussian * dev + center
    val x = Vector(D, _ => rand.nextGaussian * dev + center) 
    return DataPoint(x, y)
  }
   
  def generateData(mode: Int) :Array[DataPoint] = {
    def generatePointRandom(i: Int) = {
      val y = rand.nextDouble 
      val x = Vector(D, rand.nextDouble) 
      DataPoint(x,y)
    }
    def generatePointCentered(i: Int) = {
      if (i>0 && i < N*0.2){
        randPoint(30, 4)
      } else if (i < N*0.5){
        randPoint(3, -5)
      } else if (i < N*0.7){
        randPoint(1, 10)
      } else 
        randPoint(2, -2)
    }
    if (mode ==  1)
      Array.tabulate(N)(generatePointCentered)
    else
      Array.tabulate(N)(generatePointRandom)
  }

  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: SparkKMeans <host> [<slices>] [numIteration] [simFailure] [generator]")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "SparkKMeans")
    val numSlices = if (args.length > 1) args(1).toInt else 2
    val iter = if (args.length>2) args(2).toInt else 5
    val simFailure = if (args.length >3) args(3).toInt else 0
    println ("som failure " + simFailure)
    val gen = if (args.length >4) args(4).toInt else 0
 
    val data = generateData(gen)
    var sum = new Array[Double](iter)
    val cent = new Array[DataPoint](K)
    // init centroids
    Array.copy(data, 0, cent, 0, K)
    val centroid = Array.tabulate[Accumulator[DataPoint]](K) (x =>sc.accumulator(cent(x)))
    var last = Array.tabulate[Broadcast[DataPoint]](K)(x => sc.broadcast(centroid(x).value))
    for (j <- 0 to iter-1) {
      println("On iteration " + j)
      val count = sc.accumulator(Vector.zeros(K))
      for (parts <- sc.parallelize(data, numSlices).glom()) {
        for (p <- parts){
        if (simFailure!= 0 || (p != parts(2) || j%4 == 2))
        {
        // rewrite this with map function over cent
        var minval = (last(0).value - p).mag
        var minid = 0
        for (i <- 1 to last.length -1){
          val c = last(i)
          val dis = (c.value-p).mag()
          if (dis <= minval) {
            minval = dis
            minid = i
          }
        }
        // Add new pt to centroid accumulator
        centroid(minid) += p
        var update = Vector.zeros(K)
        update(minid) = 1
        count += update
        }}
      }
      for (i <- 0 to K -1){
        centroid(i).value = centroid(i).value * (1.0/ count.value(i))
        println ("id " + i + " value " + centroid(i).value + " count " +count.value(i))
        sum(j) += math.pow((last(i).value - centroid(i).value).mag(), 2.0)
      }

      last = Array.tabulate[Broadcast[DataPoint]](K)(x => sc.broadcast(centroid(x).value))
      
    }
      for (t <- 0 to iter-1) println ("On iteration " + t + " diff " + sum(t) )

  }
}
