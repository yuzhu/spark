package spark.examples

import java.util.Random
import scala.math.exp
import Vector._
import spark._
import spark.broadcast._
import scala.runtime.ScalaRunTime._

object SparkKMeans {
  val N = 1000  // Number of data points
  val D = 100   // Numer of dimensions
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
    def generatePointRandom(i: Int):DataPoint = {
      val y = rand.nextDouble 
      val x = Vector(D, _ => rand.nextDouble) 
      return DataPoint(x,y)
    }
    def generatePointCentered(i: Int) = {
      if (i>0 && i < N*0.2){
        randPoint(1, 4)
      } else if (i < N*0.5){
        randPoint(3, -5)
      } else if (i < N*0.7){
        randPoint(1, 10)
      } else 
        randPoint(2, -2)
    }
    if (mode ==  1)
      return Array.tabulate(N)(generatePointCentered)
    else
      return Array.tabulate(N)(generatePointRandom)
  }

  def main(args: Array[String]) {
    //./run spark.examples.SparkKMeans local[4] 4 20 0 1
    //./run spark.examples.SparkKMeans local[4] 4 20 1 1
    if (args.length == 0) {
      System.err.println("Usage: SparkKMeans <host> [<slices>] [numIteration] [simFailure] [generator]")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "SparkKMeans")
    val numSlices = if (args.length > 1) args(1).toInt else 2
    val iter = if (args.length>2) args(2).toInt else 5
    val simFailure = if (args.length >3) args(3).toInt else 0
    // println ("sim failure " + simFailure)
    val gen = if (args.length >4) args(4).toInt else 0
 
    val data = generateData(gen)
    // println("original data points")
    // data.foreach(x=>println(x)) 
    var sum = new Array[Double](iter)
    val cent = new Array[DataPoint](K)
    // init centroids
    Array.copy(data, 0, cent, 0, K)
    // last is the broadcast variable
    var last = Array.tabulate[Broadcast[DataPoint]](K)(x => sc.broadcast(cent(x)))
    var start = System.currentTimeMillis
    for (j <- 0 to iter-1) {
      println("On iteration " + j)
      val count = sc.accumulator(Vector.zeros(K))
      // last.foreach(x=>println(x.value))
      val centroid = Array.tabulate[Accumulator[DataPoint]](K) (x =>sc.accumulator(DataPoint(Vector.zeros(D), 0)))

      
      for (parts <- sc.parallelize(data, numSlices).glom()) {
        for (p <- parts){
        if ((simFailure==0) || (p != parts(2) || j%4 == 2))
        {
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
        var safecount = 0.0
        if (count.value(i) == 0)
          safecount = 1
        else
          safecount = count.value(i)
        centroid(i).value = centroid(i).value * (1.0/ safecount) 
        // println ("id " + i + " value " + centroid(i).value + " count " +count.value(i))
        println ("id " + i + " count " +count.value(i))
        sum(j) += math.pow((last(i).value - centroid(i).value).mag(), 2.0)
      }

      last = Array.tabulate[Broadcast[DataPoint]](K)(x => sc.broadcast(centroid(x).value))
      
    }
      last.foreach(x=>println(x.value))
      for (t <- 0 to iter-1) println ("On iteration " + t + " diff " + sum(t) )
      println ("Time taken: " + (System.currentTimeMillis - start))
  }
}
