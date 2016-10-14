name := "TypedMatrix"

version := "0.1"

scalaVersion := "2.11.8"

classpathTypes += "maven-plugin"

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "com.chuusai" %% "shapeless" % "2.3.2",
  "org.nd4j" % "nd4j-native-platform" % "0.4.0"
)
    