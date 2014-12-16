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

package org.apache.spark.mllib.feature

import breeze.linalg.DenseVector
import scala.collection.mutable.ListBuffer
import com.github.fommil.netlib.BLAS.{getInstance => blas}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.annotation.Experimental
import org.apache.spark.util.random.XORShiftRandom


/**
 * :: Experimental ::
 * Doc2Vec model
 */
@Experimental
class Doc2VecModel(w2v: Word2VecModel) extends Serializable {

  protected val EXP_TABLE_SIZE = 1000
  protected var maxAlpha = 0.025
  protected var minAlpha = 0.001
  protected var numIterations = 1
  protected val MAX_EXP = 6
  protected var windowSize = 8
  protected var vectorSize = 10
  protected val initRandom = new XORShiftRandom()
  protected var AVERAGE = true
  protected var HS = true
  protected var NEG_SAMPLING_SIZE = 5
  protected var DBOW = false
  protected var RANDOMIZE_WINDOW = false

  def setMaxAlpha(maxAlpha: Float): Doc2VecModel = {
    this.maxAlpha = maxAlpha
    this
  }

  def setMinAlpha(minAlpha: Float): Doc2VecModel = {
    this.minAlpha = minAlpha
    this
  }

  def setVectorSize(vectorSize: Int): Doc2VecModel = {
    this.vectorSize = vectorSize
    this
  }

  def setNumIterations(numIterations: Int): Doc2VecModel = {
    this.numIterations = numIterations
    this
  }

  def useAverage(average: Boolean): Doc2VecModel = {
    this.AVERAGE = average
    this
  }

  def useDBOW(useDBOW: Boolean): Doc2VecModel = {
    this.DBOW = useDBOW
    this
  }

  def setNegSamples(sampleSize: Int): Doc2VecModel = {
    this.NEG_SAMPLING_SIZE = sampleSize
    this
  }

  def transform(sentence: List[String]): Vector = {
    val sentenceVocab:List[VocabWord] = getVocabWords(sentence)
    if (this.DBOW) train_dbow(sentenceVocab)
    else train_dm(sentenceVocab)
  }

  def cosineSimilarity(vec1: Vector, vec2: Vector): Double = {
    val v1 = vec1.toArray
    val v2 = vec2.toArray
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.dnrm2(n, v1, 1)
    val norm2 = blas.dnrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.ddot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  private def train_dm(sentenceVocab: List[VocabWord]): Vector = {

    val pvdm = Array.fill[Float](vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    var alpha = maxAlpha

    var words: List[VocabWord] = null
    if (sentenceVocab.length < windowSize) {
      val wordBuffer = ListBuffer[VocabWord]()
      val diff = windowSize - sentenceVocab.length
      for (idx <- 0 to diff) {
        wordBuffer.append(VocabWord("NULL" + idx, 0, Array.emptyIntArray, Array.emptyIntArray, 0))
      }
      for (idx <- 0 to sentenceVocab.length-1) wordBuffer.append(sentenceVocab(idx))
      words = wordBuffer.toList
    }
    else words = sentenceVocab

    var iter = 0
    var error = 1.0
    while (!(iter > this.numIterations) && (error > 0.1)) {

      var correct = 0
      var total = 0
      var pos = 0
      var b = 0
      if (this.RANDOMIZE_WINDOW) b = initRandom.nextInt(windowSize)
      while ((pos + windowSize - 1 - b) < words.length) {
        var pvdmTmp = new DenseVector[Float](pvdm)

        var l1: DenseVector[Float] = null

        var k = pos
        val t = pos + windowSize - 1 - b
        val targetWord = words(t)
        val neu1e = new Array[Float](vectorSize)

        var outLayerNegS = Array.emptyIntArray
        if (this.NEG_SAMPLING_SIZE > 0) {
          outLayerNegS = Array.fill[Int](this.NEG_SAMPLING_SIZE)(0)
          outLayerNegS(0) = 1
        }
        
        var meanDenom = 1
        if (this.AVERAGE) {
          l1 = DenseVector.fill[Float](vectorSize)(0)
          while (k < t) {
            if (!words(k).word.contains("NULL")) {
              if (this.w2v.model.contains(words(k).word)) {
                val nextVec = new DenseVector[Float](this.w2v.model(words(k).word))
                l1 += nextVec
                meanDenom += 1
              }
            }
            k += 1
          }
          l1 += pvdmTmp
          l1 /= meanDenom.toFloat

          var d = 0
          if (this.HS) {
            while (d < targetWord.codeLen) {
              val inner = targetWord.point(d)
              val l2 = inner * vectorSize
              var f = blas.sdot(vectorSize, l1.toArray, 0, 1, w2v.syn1, l2, 1)
              total += 1
              if (f > -MAX_EXP && f < MAX_EXP) {
                val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                f = w2v.expTable(ind)
                val g = ((1 - targetWord.code(d) - f) / meanDenom.toFloat) * alpha.toFloat
                blas.saxpy(vectorSize, g, w2v.syn1, l2, 1, neu1e, 0, 1)
              }
              d += 1
            }

            d = 0
            while (d < outLayerNegS.length) {
              var inner = 0
              if (d == 0) inner = w2v.vocabHash.getOrElse(targetWord.word, -1)
              else inner = w2v.unigramTable(initRandom.nextInt(w2v.unigramTable.size))
              if (inner > -1) {
                val l2 = inner * vectorSize
                // Propagate hidden -> output
                var f = blas.sdot(vectorSize, l1.toArray, 0, 1, w2v.syn1, l2, 1)
                if (f > -MAX_EXP && f < MAX_EXP) {
                  val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                  f = w2v.expTable(ind)
                  val g = (1 - outLayerNegS(d) - f) * alpha.toFloat
                  blas.saxpy(vectorSize, g, w2v.syn1Neg, l2, 1, neu1e, 0, 1)
                }
                d += 1
              }
            }
          }
        }
        // Concatenation
        else {
          if (this.HS) {
            var d = 0
            l1 = new DenseVector[Float](this.w2v.model.getOrElse(words(k).word, Array.empty))
            while (d < targetWord.codeLen && l1.length > 0) {
              val inner = targetWord.point(d)
              val l2 = inner * vectorSize
              var f = blas.sdot(vectorSize, pvdmTmp.toArray, 0, 1, w2v.syn1, l2, 1)
              total += 1
              if (f > -MAX_EXP && f < MAX_EXP) {
                val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                f = w2v.expTable(ind)
                val err = targetWord.code(d) - f
                if (math.abs(err) < 0.5) correct += 1
                alpha = math.max(
                  alpha - (maxAlpha - minAlpha) / (numIterations.toFloat * words.length),
                  this.minAlpha
                )
                val g = ((1 - targetWord.code(d) - f) / meanDenom.toFloat) * alpha.toFloat
                blas.saxpy(vectorSize, g, w2v.syn1, l2, 1, neu1e, 0, 1)
              }
              d += 1
            }

            while (k < words.length && k < t) {
              d = 0
              while (d < targetWord.codeLen && l1.length > 0) {
                val inner = targetWord.point(d)
                val l2 = inner * vectorSize
                var f = blas.sdot(vectorSize, l1.toArray, 0, 1, w2v.syn1, l2, 1)
                total += 1
                if (f > -MAX_EXP && f < MAX_EXP) {
                  val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                  f = w2v.expTable(ind)
                  val err = targetWord.code(d) - f
                  if (math.abs(err) < 0.5) correct += 1
                  alpha = math.max(
                    alpha - (maxAlpha - minAlpha) / (numIterations.toFloat * words.length),
                    this.minAlpha
                  )
                  val g = ((1 - targetWord.code(d) - f) / meanDenom.toFloat) * alpha.toFloat
                  blas.saxpy(vectorSize, g, w2v.syn1, l2, 1, neu1e, 0, 1)
                }
                d += 1
              }
              k += 1
            }
          }
          if (this.NEG_SAMPLING_SIZE > 0) {
            // TODO
          }
        }
        blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, pvdm, 0, 1)

        alpha = math.max(
          alpha - (maxAlpha - minAlpha) / (numIterations.toFloat * words.length),
          this.minAlpha)

        pos += 1
      }
      error =  1d - correct.toDouble / total.toDouble
      iter += 1
    }
    Vectors.dense(pvdm.map(_.toDouble))
  }

  private def train_dbow(sentenceVocab: List[VocabWord]): Vector = {

    val pvdm = Array.fill[Float](vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    var alpha = maxAlpha

    var words: List[VocabWord] = null
    if (sentenceVocab.length < windowSize) {
      val wordBuffer = ListBuffer[VocabWord]()
      val diff = sentenceVocab.length - windowSize
      for (idx <- 0 to diff + 1) {
        wordBuffer.append(VocabWord("NULL" + idx, 0, Array.emptyIntArray, Array.emptyIntArray, 0))
      }
      for (idx <- 0 to sentenceVocab.length-1) wordBuffer.append(sentenceVocab(idx))
      words = wordBuffer.toList
    }
    else words = sentenceVocab

    var iter = 0
    var error = 1.0
    while (!(iter > this.numIterations) && (error > 0.1)) {

      var correct = 0
      var total = 0
      var pos = 0
      val b = initRandom.nextInt(windowSize)
      while ((pos + windowSize - 1 - b) < words.length) {

        val neu1e = new Array[Float](vectorSize)
        for (s <- 0 to windowSize - b + 1) {
          val t = pos + initRandom.nextInt(windowSize - 1)
          if (t < words.length) {
            val targetWord = words(t)
            var d = 0
            if (this.HS) {
              while (d < targetWord.codeLen) {
                val inner = targetWord.point(d)
                val l2 = inner * vectorSize
                var f = blas.sdot(vectorSize, pvdm, 0, 1, w2v.syn1, l2, 1)
                total += 1
                if (f > -MAX_EXP && f < MAX_EXP) {
                  val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                  f = w2v.expTable(ind)
                  val err = targetWord.code(d) - f
                  if (math.abs(err) < 0.5) correct += 1
                  alpha = math.max(
                    alpha - (maxAlpha - minAlpha) / (numIterations.toFloat * words.length),
                    this.minAlpha)
                  val g = (1 - err * alpha).toFloat
                  blas.saxpy(vectorSize, g, w2v.syn1, l2, 1, neu1e, 0, 1)
                }
                d += 1
              }
            }
          }
          if (this.NEG_SAMPLING_SIZE > 0) {
            // TODO
          }
        }

        blas.saxpy(vectorSize, 1.0f / (windowSize.toFloat + 1.0f), neu1e, 0, 1, pvdm, 0, 1)
        pos += 1
      }
      error =  1d - correct.toDouble / total.toDouble
      iter += 1
    }
    Vectors.dense(pvdm.map(_.toDouble))
  }

  private def getVocabWords(words: List[String]): List[VocabWord] = {
    val vocabWords: ListBuffer[VocabWord] = ListBuffer[VocabWord]()
    for (word <- words) {
      val idx = w2v.vocabHash.getOrElse(word, -1)
      if (idx >= 0) vocabWords.append(w2v.vocab(idx))
    }
    vocabWords.toList
  }
}
