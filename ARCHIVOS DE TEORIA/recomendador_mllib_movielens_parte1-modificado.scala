// iniciar sesión como $> spark-shell --driver-memory 4g

// cargamos las partes de Spark que necesitamos
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

// import org.apache.spark.mllib.recommendation.


val datosBruto=sc.textFile("/home/usuario/tescr/reducido.txt")
// comprobamos que el primer registro esté cargado
datosBruto.first()
// Ejemplo de salida>> res0: String = 196	242	3	881250949

// formato del archivo de datos: id_usuario id_pelicula rating timestamp separados por \t
val ratingsBruto = datosBruto.map(_.split("\t"))

// observamos los primeros dos registros 
//
ratingsBruto.take(2)
// Ejemplo de salida >> res2: Array[Array[String]] = Array(Array(196, 242, 3, 881250949), Array(186, 302, 3, 891717742))

// Los métodos de un objeto de tipo Recomendador ALS serían:
//ALS.
//<tabulador>
// <console>:25: error: ambiguous reference to overloaded definition,
// both method train in object ALS of type (ratings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating], rank: Int, iterations: Int)org.apache.spark.mllib.recommendation.MatrixFactorizationModel
// and  method train in object ALS of type (ratings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating], rank: Int, iterations: Int, lambda: Double)org.apache.spark.mllib.recommendation.MatrixFactorizationModel
// los importantes para nosotros son: .train y .trainImplicit
// ALS train requiere: (ratings: RRD[Rating], rank: Int, iterations: Int)
// ó bien: (ratings: RRD[Rating], rank: Int, iterations: Int, lambda: Double)

// Transformamos cada Array[String] de ratingsBruto en un RDD de Ratings
val ratings=ratingsBruto.map{ case Array(id_usuario, id_pelicula, rating, _) => // no nos interesa timestamp
Rating(id_usuario.toInt, id_pelicula.toInt, rating.toDouble)}

// observamos el primer elemento
println("Elementos convertidos:")
val primerosCincoElementos = ratings.take(5)
primerosCincoElementos.foreach(println)
println("\n")
// Ejemplo de salida: res5: org.apache.spark.mllib.recommendation.Rating = Rating(196,242,3.0)

// PREPARADOS PARA COMENZAR EL ENTRENAMIENTO

val rank=10 // se refiere al número de características latentes en las matrices U e I: 
// U es |Usuarios| x rank e I = rank x |Items| 
// Depende del problema. Se suele ubicar entre 10 y 200

val iterations=10 // número de iteraciones para converger. ALS garantiza que termina y mejora en cada iteración. 

val lambda=0.01 // regula el sobre-ajuste. Debería ajustarse en las etapas de aprendizaje usando por ej. validación cruzada

// hacemos aprender al modelo
val modelo=ALS.train(ratings, rank, iterations, lambda)

// el modelo es
// Ejemplo de salida: modelo: org.apache.spark.mllib.recommendation.MatrixFactorizationModel
// Los objetos MatrixFactorizationModel contienen userFeatures y productFeatures como pares (id, factor)

modelo.userFeatures.count
modelo.productFeatures.count


