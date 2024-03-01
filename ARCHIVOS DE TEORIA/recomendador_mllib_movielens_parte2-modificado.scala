
// AQUÍ COMIENZA EL USO DEL MODELO DE RECOMENDADOR
// tenemos que tener en el caso de los usuarios 943 elementos, 1 por usuario, y 1682 para los productos, las películas

val ratingPredicho=modelo.predict(186, 222)
// ejemplo de salida: ratingPredicho: Double = 4.394692426495602

// calcular la lista de recomendaciones top-k para un usuario
val id_usuario=186
val k=10
val topKRecs=modelo.recommendProducts(id_usuario, k)
// topKRecs: Array[org.apache.spark.mllib.recommendation.Rating] = Array(Rating(789,56,5.755155863395549), Rating(789,185,5.373799474542957), Rating(789,502,5.295197428631495), Rating(789,42,5.283351735311406), Rating(789,346,5.19606815903618), Rating(789,169,5.193933187081189), Rating(789,23,5.166246921666904), Rating(789,429,5.151600439919473), Rating(789,198,5.146039377387927), Rating(789,108,5.10298689507899))

println(topKRecs.mkString("\n"))
// Rating(789,56,5.755155863395549)
// Rating(789,185,5.373799474542957)
// Rating(789,502,5.295197428631495)
// Rating(789,42,5.283351735311406)
// Rating(789,346,5.19606815903618)
// Rating(789,169,5.193933187081189)
// Rating(789,23,5.166246921666904)
// Rating(789,429,5.151600439919473)
// Rating(789,198,5.146039377387927)
// Rating(789,108,5.10298689507899)

// para mirar las recomendaciones primero analizamos las peliculas que ha calificado el usuario 123
val peliculas = sc.textFile("/home/usuario/tescr/items.txt")
val titulos = peliculas.map(_.split('|').take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
titulos(123)

// ratings o calificaciones para el usuario
val peliculasParaUsuario = ratings.keyBy(_.user).lookup(id_usuario)
println("Numero de peliculas recomendadas para usuario: " + peliculasParaUsuario.size)  // numero de peliculas que ha calificado

// sacamos las 10 primeras del ránking
println("Peliculas para el usuario: \n")
peliculasParaUsuario.sortBy(-_.rating).take(10).map{rating => (titulos(rating.product), rating.rating)}.foreach(println)

// 10 primeras recomendaciones para el usuario
println("Titulos de Peliculas para el usuario: \n")
topKRecs.map{rating => (titulos(rating.product), rating.rating)}.foreach(println)

