/*
    Master en Ingeniería Informática - Universidad de Valladolid

    TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
    Mini-Proyecto 1 : Recomendadores

    Dataset: CDs and Vinyl

    Primer script

    Grupo 2:
    Sergio Agudelo Bernal
    Miguel Ángel Collado Alonso
    José María Lozano Olmedo
*/

// Arrancar spark-shell como: $ spark-shell --driver-memory 8g


// Importación de módulos a usar
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Calendar
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

// Variables de ruta del archivo de datos
val PATH = "/home/usuario/Regresion/"
val ARCHIVO_TEST = "conjunto-test"
val ARCHIVO_MODELO = "modeloALS"

// ----------------------------------------------------------------------------------------
// Carga del conjunto de pruebas
// ----------------------------------------------------------------------------------------

println("\nCARGA DEL CONJUNTO DE PRUEBAS")

// case class para poder mover Ratings y usar los nombres de sus campos
case class Rating(itemId: Int, userId: Int, rating: Float)

// a partir de un string genera un objeto Rating()
def parseRating(str: String): Rating = {
  val fields = str.split(",")
  assert(fields.size == 3)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
}

// leemos los datos y los convertimos en un DataFrame
val test = spark.read.textFile(PATH + ARCHIVO_TEST).map(parseRating).toDF()

test.show(5)

// ----------------------------------------------------------------------------------------
// Carga del modelo
// ----------------------------------------------------------------------------------------

println("\nCARGA DEL MODELO")

val model = ALSModel.load(PATH + ARCHIVO_MODELO)

// ----------------------------------------------------------------------------------------
// Evaluación del modelo
// ----------------------------------------------------------------------------------------

println("\nEVALUACIÓN DEL MODELO")

// Configuración de estrategia y generación de predicciones
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

// Cálculo del error cuadrático medio
val rmse = evaluator.evaluate(predictions)
println(s"Error cuadrático medio: $rmse")

// ----------------------------------------------------------------------------------------
// Generación de recomendaciones
// ----------------------------------------------------------------------------------------

// // Obtenemos 3 recomendaciones para cada usuario
// val userRecs = model.recommendForAllUsers(3)
// println(userRecs.getClass)
// userRecs.show()
// // Seleccionamos 3 recomendaciones de usuarios para cada item
// val itemRecs = model.recommendForAllItems(3)
// println(itemRecs.getClass)
// itemRecs.show()

// // Generamos tres recomendaciones de items para un conjunto de usuarios
// val users = ratings.select(als.getUserCol).distinct().limit(3)
// println(users.getClass)
// users.show()
// val userSubsetRecs = model.recommendForUserSubset(users, 3)
// println(userSubsetRecs.getClass)
// userSubsetRecs.show()