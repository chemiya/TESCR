/*
    Master en Ingeniería Informática - Universidad de Valladolid

    TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
    Mini-Proyecto 1 : Recomendadores

    Dataset: CDs and Vinyl

    Segundo script

    Grupo 2:
    Sergio Agudelo Bernal
    Miguel Ángel Collado Alonso
    José María Lozano Olmedo
*/

// Arrancar spark-shell como: $ spark-shell --driver-memory 8g --executor-memory 8g --executor-cores 4

// Indispensable haber ejecutado el primes script!

// Configuraciones iniciales
sc.setCheckpointDir("checkpoint")
sc.setLogLevel("ERROR")

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
case class Rating(
    itemId: Int,
    item: String,
    userId: Int,
    user: String,
    rating: Float,
    timestamp: Int
)

case class ItemDict(
    item: String,
    itemId: Int
)

case class UserDict(
    user: String,
    userId: Int
)

// a partir de un string genera un objeto Rating()
def parseRating(str: String): Rating = {
    val fields = str.split(",")
    assert(fields.size == 6)

    Rating(
        fields(0).toInt,
        fields(1).toString,
        fields(2).toInt,
        fields(3).toString,
        fields(4).toFloat,
        fields(5).toInt
    )
}

def parseUserLookup(str: String): UserDict = {
    val fields = str.split(",")
    assert(fields.size == 2)

    UserDict(
        fields(0).toString,
        fields(1).toInt
    )
}

def parseItemLookup(str: String): ItemDict = {
    val fields = str.split(",")
    assert(fields.size == 2)

    ItemDict(
        fields(0).toString,
        fields(1).toInt
    )
}

// leemos los datos y los convertimos en un DataFrame
val test = spark.read.textFile(PATH + ARCHIVO_TEST).map(parseRating).toDF()

test.show(5)

// Diccionarios de lookup para obtener el user e item a partir de sus Ids
// Indispensable haber ejecutado el primes script!
val dfUserLookup = spark.read.textFile(PATH + "dfUserLookup").map(parseUserLookup).toDF()
val dfItemLookup = spark.read.textFile(PATH + "dfItemLookup").map(parseItemLookup).toDF()

// ----------------------------------------------------------------------------------------
// Carga del modelo
// ----------------------------------------------------------------------------------------

println("\nCARGA DEL MODELO")

val model = ALSModel.load(PATH + ARCHIVO_MODELO)

// ----------------------------------------------------------------------------------------
// Evaluación del modelo
// ----------------------------------------------------------------------------------------

println("\nEVALUACIÓN DEL MODELO")

// Creación de evaluador por métrica RMSE
val evaluator = new RegressionEvaluator().
    setMetricName("rmse").
    setLabelCol("rating").
    setPredictionCol("prediction")

// Configuración de estrategia y generación de predicciones
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

// Cálculo del error cuadrático medio
val rmse = evaluator.evaluate(predictions)
println(s"Error cuadrático medio: $rmse")

// ----------------------------------------------------------------------------------------
// Generación de recomendaciones
// ----------------------------------------------------------------------------------------

println("\nGENERACIÓN DE RECOMENDACIONES")

// Generamos tres recomendaciones de items para un conjunto de usuarios
val users = test.select(model.getUserCol).distinct().limit(10)
println(users.getClass)

val userSubsetRecs = model.recommendForUserSubset(users, 3)
println(userSubsetRecs.getClass)
userSubsetRecs.show()

// Selección de usuario de ejemplo para visualizar recomendaciones
val usuario = 188614

println(s"Usuario:")
test.filter(col("userId") === usuario).show()

// Visualización de recomendaciones
println("Recomendaciones:")
userSubsetRecs.filter(col("userId") === usuario).take(1)

// Detalle de recomendaciones
dfItemLookup.
    filter(col("itemId") === 262006 || col("itemId") === 210566 || col("itemId") === 265556).
    select("itemId", "item").
    distinct().
    show()