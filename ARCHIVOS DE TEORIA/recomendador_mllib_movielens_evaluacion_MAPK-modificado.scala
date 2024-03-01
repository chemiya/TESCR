// calculo de la precisión promedio
def avgPrecisionK(real: Seq[Int], predichos: Seq[Int], k: Int): Double = {
 val predk= predichos.take(k)
 var score = 0.0
 var aciertos = 0.0
 for ((p, i) <- predk.zipWithIndex) {
   if(real.contains(p)) {
    aciertos += 1.0
    score += aciertos / (i.toDouble + 1.0) 
   }
 }
 if (real.isEmpty) {
  1.0
 } else {
  score / scala.math.min(real.size, k).toDouble
 }
}

// ejemplo para un usuario concreto
val peliculasReales = peliculasParaUsuario.map(_.product)
val peliculasPredichas = topKRecs.map(_.product)
val apk10=avgPrecisionK(peliculasReales, peliculasPredichas, 10)
// posible salida ==> apk10: Double = 0.0


// 10 primeras recomendaciones de productos por usuario
val topkRecsTodos = modelo.recommendProductsForUsers(10)

// obtengo el formato para cada usuario: (id_usuario, recs) donde recs es un Seq[Int] de recomendaciones por usuario
val allRecs=topkRecsTodos.map{ case (id_usuario, rates) =>  
  val items = rates.map(rate  => rate.product)
  (id_usuario, items.toSeq)
}

val peliculasUsuarioReales= ratings.map{ case Rating(user, product, rating) => (user, product)}.groupBy(_._1)

val K=10
// salida K: Int = 10

val MAPK=allRecs.join(peliculasUsuarioReales).map{
     case (id_usuario, (predichas, realesConIds)) =>
       val reales=realesConIds.map(_._2).toSeq
       avgPrecisionK(reales, predichas, K)
   }.reduce(_ + _) / allRecs.count
// posible salida: MAPK: Double = 0.07072619468430719                                              

println("Mean Average Precision at K = " + MAPK)
// posible salida: Mean Average Precision at K = 0.07072619468430719

// METODO MAPK IMPLEMENTADO EN MLLIB
// necesita allRecs, peliculasUsuarioReales del apartado anterior
import org.apache.spark.mllib.evaluation.RankingMetrics
val predictedAndTrueRanking = allRecs.join(peliculasUsuarioReales).map {
  case (id_usuario, (predichas, realesConIds)) => 
     val reales = realesConIds.map(_._2)
     (predichas.toArray, reales.toArray)
}
val rankingMetrics = new RankingMetrics(predictedAndTrueRanking)
println("Mean Average Prediction at K= " + rankingMetrics.meanAveragePrecision)

// LA diferencia entre ambos valores se debe a que RankingMetrics usa un valor K muy alto, mucho más que el que hemos uado
val MAPK2000=allRecs.join(peliculasUsuarioReales).map{
          case (id_usuario, (predichas, realesConIds)) =>
            val reales=realesConIds.map(_._2).toSeq
            avgPrecisionK(reales, predichas, 2000)
     }.reduce(_ + _) / allRecs.count
// posible salida MAPK2000: Double = 0.0075104817599333075
// con K=500  MAPK500: Double = 0.007527631070957529



