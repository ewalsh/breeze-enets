name := "enets"

version := "1.03"

scalaVersion := "2.12.8"

libraryDependencies  ++= Seq(
  // Last stable release
"org.scalanlp" %% "breeze" % "1.0",

// Native libraries are not included by default. add this if you want them (as of 0.7)
// Native libraries greatly improve performance, but increase jar sizes.
// It also packages various blas implementations, which have licenses that may or may not
// be compatible with the Apache License. No GPL code, as best I know.
"org.scalanlp" %% "breeze-natives" % "1.0",

// The visualization library is distributed separately as well.
// It depends on LGPL code
"org.scalanlp" %% "breeze-viz" % "1.0",

// To utilize parallel processing within the cross-validation model
// this fork adds the akka framework
"com.typesafe.akka" %% "akka-actor" % "2.6.11"
)

scalacOptions ++= Seq("-deprecation", "-feature")

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
