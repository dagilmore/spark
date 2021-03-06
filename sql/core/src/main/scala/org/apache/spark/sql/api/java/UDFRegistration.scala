/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package org.apache.spark.sql.api.java

import org.apache.spark.sql.catalyst.expressions.{Expression, ScalaUdf}
import org.apache.spark.sql.types.util.DataTypeConversions._

/**
 * A collection of functions that allow Java users to register UDFs.  In order to handle functions
 * of varying airities with minimal boilerplate for our users, we generate classes and functions
 * for each airity up to 22.  The code for this generation can be found in comments in this trait.
 */
private[java] trait UDFRegistration {
  self: JavaSQLContext =>

  /* The following functions and required interfaces are generated with these code fragments:

   (1 to 22).foreach { i =>
     val extTypeArgs = (1 to i).map(_ => "_").mkString(", ")
     val anyTypeArgs = (1 to i).map(_ => "Any").mkString(", ")
     val anyCast = s".asInstanceOf[UDF$i[$anyTypeArgs, Any]]"
     val anyParams = (1 to i).map(_ => "_: Any").mkString(", ")
     println(s"""
         |def registerFunction(
         |    name: String, f: UDF$i[$extTypeArgs, _], @transient dataType: DataType) = {
         |  val scalaType = asScalaDataType(dataType)
         |  sqlContext.functionRegistry.registerFunction(
         |    name,
         |    (e: Seq[Expression]) => ScalaUdf(f$anyCast.call($anyParams), scalaType, e))
         |}
       """.stripMargin)
   }

  import java.io.File
  import org.apache.spark.sql.catalyst.util.stringToFile
  val directory = new File("sql/core/src/main/java/org/apache/spark/sql/api/java/")
  (1 to 22).foreach { i =>
    val typeArgs = (1 to i).map(i => s"T$i").mkString(", ")
    val args = (1 to i).map(i => s"T$i t$i").mkString(", ")

    val contents =
      s"""/*
         | * Licensed to the Apache Software Foundation (ASF) under one or more
         | * contributor license agreements.  See the NOTICE file distributed with
         | * this work for additional information regarding copyright ownership.
         | * The ASF licenses this file to You under the Apache License, Version 2.0
         | * (the "License"); you may not use this file except in compliance with
         | * the License.  You may obtain a copy of the License at
         | *
         | *    http://www.apache.org/licenses/LICENSE-2.0
         | *
         | * Unless required by applicable law or agreed to in writing, software
         | * distributed under the License is distributed on an "AS IS" BASIS,
         | * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         | * See the License for the specific language governing permissions and
         | * limitations under the License.
         | */
         |
         |package org.apache.spark.sql.api.java;
         |
         |import java.io.Serializable;
         |
         |// **************************************************
         |// THIS FILE IS AUTOGENERATED BY CODE IN
         |// org.apache.spark.sql.api.java.FunctionRegistration
         |// **************************************************
         |
         |/**
         | * A Spark SQL UDF that has $i arguments.
         | */
         |public interface UDF$i<$typeArgs, R> extends Serializable {
         |  public R call($args) throws Exception;
         |}
         |""".stripMargin

      stringToFile(new File(directory, s"UDF$i.java"), contents)
  }

  */

  // scalastyle:off
  def registerFunction(name: String, f: UDF1[_, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF1[Any, Any]].call(_: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF2[_, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF2[Any, Any, Any]].call(_: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF3[_, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF3[Any, Any, Any, Any]].call(_: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF4[_, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF4[Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF5[_, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF5[Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF6[_, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF6[Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF7[_, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF7[Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF8[_, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF8[Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF9[_, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF9[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF10[_, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF10[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF11[_, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF11[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF12[_, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF12[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF13[_, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF13[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF14[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF14[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF15[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF15[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF16[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF16[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF17[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF17[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF18[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF18[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF19[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF19[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF20[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF20[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF21[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF21[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  def registerFunction(name: String, f: UDF22[_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _], dataType: DataType) = {
    val scalaType = asScalaDataType(dataType)
    sqlContext.functionRegistry.registerFunction(
      name,
      (e: Seq[Expression]) => ScalaUdf(f.asInstanceOf[UDF22[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]].call(_: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any, _: Any), scalaType, e))
  }

  // scalastyle:on
}
