import breeze.linalg._

def cosineSimilarity(v1:DenseVector[Double], v2:DenseVector[Double]):Double = {
(v1 dot v2)/(norm(v1)*norm(v2))
}

val itemId=186

val itemFactor=modelo.productFeatures.lookup(itemId).head

val itemVector=new DenseVector(itemFactor)

cosineSimilarity(itemVector, itemVector)

val similares=modelo.productFeatures.map{
 case (id, factor) => 
  val factorVector = new DenseVector(factor)
  val similitud = cosineSimilarity(factorVector, itemVector)
  (id, similitud)
}

val K=10
val similitudesOrdenadas=similares.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity})

println(similitudesOrdenadas.take(K+1).mkString("\n"))

// cogemos los titulos

val pelisBruto=sc.textFile("/home/usuario/tescr/items.txt")
val titulosRDD=pelisBruto.map{_.split("\\|")}.map(strings => strings(1))
val Titulos= titulosRDD.collect()

// sacamos los titulos de las peliculas más similares
val pelisSimilares=similares.top(K)(Ordering.by[(Int, Double), Double] {
  case (id, similarity) => similarity})

pelisSimilares.slice(1,11).map{case (id, sim) => (Titulos(id-1), sim)}.mkString("\n")



