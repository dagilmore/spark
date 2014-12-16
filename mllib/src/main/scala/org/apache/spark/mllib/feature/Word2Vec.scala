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

import java.lang.{Iterable => JavaIterable}

import breeze.linalg.DenseVector
import org.apache.spark.broadcast.Broadcast

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd._
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

/**
 * Entry in vocabulary
 */
protected case class  VocabWord(
                                var word: String,
                                var cn: Int,
                                var point: Array[Int],
                                var code: Array[Int],
                                var codeLen: Int
                                )

/**
 * :: Experimental ::
 * Word2Vec creates vector representation of words in a text corpus.
 * The algorithm first constructs a vocabulary from the corpus
 * and then learns vector representation of words in the vocabulary.
 * The vector representation can be used as features in
 * natural language processing and machine learning algorithms.
 *
 * We used skip-gram model in our implementation and hierarchical softmax
 * method to train the model. The variable names in the implementation
 * matches the original C implementation.
 *
 * For original C implementation, see https://code.google.com/p/word2vec/
 * For research papers, see
 * Efficient Estimation of Word Representations in Vector Space
 * and
 * Distributed Representations of Words and Phrases and their Compositionality.
 */
@Experimental
class Word2Vec extends Serializable with Logging {

  protected var vectorSize = 100
  protected var learningRate = 0.025
  protected var numPartitions = 1
  protected var numIterations = 1
  protected var seed = Utils.random.nextLong()
  protected var TRAIN_SG = true
  protected var TRAIN_CBOW = false
  protected var PREDICT_MIDDLE = true
  protected var USE_HIERARCHICAL_SOFTMAX = true

  /**
   * Sets vector size (default: 100).
   */
  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }

  /**
   * Sets the unigram distribution table size (default 100000000)
   */
  def setTableSize(tableSize: Int): this.type = {
    this.TABLE_SIZE = tableSize
    this
  }

  /**
   * Sets the number of negative samples to perform (default 10)
   */
  def setNegativeSamples(sampleSize: Int): this.type = {
    this.NEG_SAMPLING_SIZE = sampleSize
    this
  }

  /**
   * Set whether to use hierarchical softmax or not
   */
  def setHierarchicalSoftmax(useHS: Boolean): this.type = {
    this.USE_HIERARCHICAL_SOFTMAX = useHS
    this
  }

  /**
   * Sets initial learning rate (default: 0.025).
   */
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  /**
   * Sets number of partitions (default: 1). Use a small number for accuracy.
   */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0, s"numPartitions must be greater than 0 but got $numPartitions")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */
  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */
  def setSkipgram(skipgram: Boolean): this.type = {
    this.SKIPGRAM = skipgram
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }

  /**
   * Sets random seed (default: a random long integer).
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Switch between predict middle and predict next architectures
   */
  def setPredictMiddle(predictMiddle: Boolean): this.type = {
    this.PREDICT_MIDDLE = predictMiddle
    this
  }

  /**
   * Switch between average and concatenation network architectures
   */
  def useAverage(average: Boolean): this.type = {
    this.AVERAGE = average
    this
  }

  protected val EXP_TABLE_SIZE = 1000
  protected val MAX_EXP = 6
  protected val MAX_CODE_LENGTH = 40
  protected val MAX_SENTENCE_LENGTH = 1000
  protected var SKIPGRAM = true
  protected var NEG_SAMPLING_SIZE = 5
  protected var TABLE_SIZE = 100000000
  protected var table: Array[Int] = Array.emptyIntArray
  protected var AVERAGE = true

  /** context words from [-window, window] */
  protected val window = 5

  /** minimum frequency to consider a vocabulary word */
  protected var minCount = 5

  protected var trainWordsCount = 0
  protected var vocabSize = 0
  protected var vocab: Array[VocabWord] = null
  protected var vocabHash = mutable.HashMap.empty[String, Int]

  protected def learnVocab(words: RDD[String]): Unit = {
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .map(x => VocabWord(
      x._1,
      x._2,
      new Array[Int](MAX_CODE_LENGTH),
      new Array[Int](MAX_CODE_LENGTH),
      0))
      .filter(_.cn >= minCount)
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    logInfo("trainWordsCount = " + trainWordsCount)
  }

  protected def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  protected def initUnigramTable(): Unit = {

    var a, i = 0
    var trainWordsPow = 0d
    val power = .75d
    var d1 = power
    this.table = new Array[Int](this.TABLE_SIZE)
    while (a < vocabSize) {
      trainWordsPow += math.pow(this.vocab(a).cn.toDouble, power);
      a += 1
    }
    i = 0
    d1 = math.pow(this.vocab(i).cn.toDouble, power) / trainWordsPow
    a = 0
    while (a < this.TABLE_SIZE) {
      this.table(a) = i
      if (a.toDouble / this.TABLE_SIZE.toDouble > d1) {
        i += 1
        d1 += math.pow(this.vocab(i).cn.toDouble, power) / trainWordsPow
      }
      a += 1
    }
  }

  protected def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 + 1)
    val binary = new Array[Int](vocabSize * 2 + 1)
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    val code = new Array[Int](MAX_CODE_LENGTH)
    val point = new Array[Int](MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   * @param dataset an RDD of words
   * @return a Word2VecModel
   */
  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {

    val words = dataset.flatMap(x => x)

    learnVocab(words)
    createBinaryTree()
    if (this.NEG_SAMPLING_SIZE > 0) initUnigramTable()

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)

    val sentences: RDD[Array[Int]] = words.mapPartitions { iter =>
      new Iterator[Array[Int]] {
        def hasNext: Boolean = iter.hasNext

        def next(): Array[Int] = {
          var sentence = new ArrayBuffer[Int]
          var sentenceLength = 0
          while (iter.hasNext && sentenceLength < MAX_SENTENCE_LENGTH) {
            val word = bcVocabHash.value.get(iter.next())
            word match {
              case Some(w) =>
                sentence += w
                sentenceLength += 1
              case None =>
            }
          }
          sentence.toArray
        }
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)
    val syn0Global =
      Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    val syn1NegGlobal = new Array[Float](vocabSize * vectorSize)
    var alpha = learningRate
    for (k <- 1 to numIterations) {
      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = new Array[Int](vocabSize)
        val syn1NegModify = new Array[Int](vocabSize)
        val model = iter.foldLeft((syn0Global, syn1Global, syn1NegGlobal, 0, 0)) {
          case ((syn0, syn1, syn1Neg, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              // TODO: discount by iteration?
              alpha =
                learningRate * (1 - numPartitions * wordCount.toDouble / (trainWordsCount + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              logInfo("wordCount = " + wordCount + ", alpha = " + alpha)
            }
            wc += sentence.size

            // Skip gram training
            if (this.SKIPGRAM) {
              var pos = 0
              var outLayerNegS = Array.emptyIntArray
              if (this.NEG_SAMPLING_SIZE > 0) {
                outLayerNegS = Array.fill[Int](this.NEG_SAMPLING_SIZE)(0)
                outLayerNegS(0) = 1
              }
              while (pos < sentence.size) {
                val word = sentence(pos)
                val b = random.nextInt(window)
                var a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = pos - window + a
                    if (c >= 0 && c < sentence.size) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      val neu1e = new Array[Float](vectorSize)

                      // Hierarchical softmax
                      var d = 0
                      if (this.USE_HIERARCHICAL_SOFTMAX) {
                        while (d < bcVocab.value(word).codeLen) {
                          val inner = bcVocab.value(word).point(d)
                          val l2 = inner * vectorSize
                          // Propagate hidden -> output
                          var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                          if (f > -MAX_EXP && f < MAX_EXP) {
                            val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                            f = expTable.value(ind)
                            val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                            blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                            blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                            syn1Modify(inner) += 1
                          }
                          d += 1
                        }
                      }

                      // Negative Sampling
                      d = 0
                      while (d < outLayerNegS.length) {
                        var inner = 0
                        if (d == 0) inner = sentence(pos)
                        else inner = this.table(random.nextInt(this.TABLE_SIZE))
                        val l2 = inner * vectorSize
                        // Propagate hidden -> output
                        var f = blas.sdot(vectorSize, syn0, l1, 1, syn1Neg, l2, 1)
                        if (f > -MAX_EXP && f < MAX_EXP) {
                          val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                          f = expTable.value(ind)
                          val g = (1 - outLayerNegS(d) - f) * alpha.toFloat
                          blas.saxpy(vectorSize, g, syn1Neg, l2, 1, neu1e, 0, 1)
                          blas.saxpy(vectorSize, g, syn0, l1, 1, syn1Neg, l2, 1)
                          syn1NegModify(inner) += 1
                        }
                        d += 1
                      }

                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                      syn0Modify(lastWord) += 1
                    }
                  }
                  a += 1
                }
                pos += 1
              }
            }
            // CBOW Training
            else {
              var pos = 0
              while (pos < sentence.length) {
                var k = pos
                val t = pos + window + 1
                var layer1 = DenseVector.fill[Float](vectorSize)(0)
                val neu1e = new Array[Float](vectorSize)
                if (t < sentence.length) {
                  if (this.AVERAGE) {
                    // TODO: Change if predicting next
                    while (k < t + window) {
                      if (k < sentence.length && k != t) {
                        val beg: Int = sentence(k) * vectorSize
                        val end: Int = (sentence(k) + 1) * vectorSize
                        layer1 += new DenseVector[Float](syn0.slice(beg, end))
                      }
                      k += 1
                    }
                    layer1 /= window.toFloat
                    var d = 0
                    val targetWord = bcVocab.value(sentence(t))
                    while (d < targetWord.codeLen) {
                      val inner = targetWord.point(d)
                      val l2 = inner * vectorSize
                      var f = blas.sdot(vectorSize, layer1.toArray, 0, 1, syn1, l2, 1)
                      if (f > -MAX_EXP && f < MAX_EXP) {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)).toInt
                        f = expTable.value(ind)
                        val g = (1 - targetWord.code(d) - f * alpha).toFloat
                        blas.saxpy(vectorSize, g / (window.toFloat + 1.0f),
                          syn1, l2, 1, neu1e, 0, 1)
                      }
                      d += 1
                    }
                    k = pos
                    while (k < t + window) {
                      if (k < sentence.length && k != t) {
                        blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, sentence(k) * vectorSize, 1)
                        syn0Modify(sentence(k)) += 1
                      }
                      k += 1
                    }
                  }
                }
                pos += 1
              }
            }

            (syn0, syn1, syn1Neg, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        val syn1NegLocal = model._3
        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1NegModify(index) > 0) {
            Some((index + vocabSize * 2,
              syn1NegLocal.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }
      val synAgg = partial.reduceByKey { case (v1, v2) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        v1
      }.collect()
      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        }
        else if (index < vocabSize * 2) {
          Array.copy(synAgg(i)._2, 0, syn1Global,
            (index - vocabSize) * vectorSize, vectorSize)
        }
        else {
          Array.copy(synAgg(i)._2, 0, syn1NegGlobal,
            (index - (2 * vocabSize)) * vectorSize, vectorSize)
        }
        i += 1
      }
    }
    newSentences.unpersist()

    val word2VecMap = mutable.HashMap.empty[String, Array[Float]]
    var i = 0
    while (i < vocabSize) {
      val word = bcVocab.value(i).word
      val vector = new Array[Float](vectorSize)
      Array.copy(syn0Global, i * vectorSize, vector, 0, vectorSize)
      word2VecMap += word -> vector
      i += 1
    }

    new Word2VecModel(word2VecMap.toMap, vocab, vocabHash,
      syn0Global, syn1Global, syn1NegGlobal, expTable.value, table)
  }

  /**
   * Computes the vector representation of each word in vocabulary (Java version).
   * @param dataset a JavaRDD of words
   * @return a Word2VecModel
   */
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): Word2VecModel = {
    fit(dataset.rdd.map(_.asScala))
  }
}

/**
 * :: Experimental ::
 * Word2Vec model
 */
@Experimental
class Word2VecModel protected[mllib](
                                    val model: Map[String, Array[Float]],
                                    val vocab: Array[VocabWord],
                                    val vocabHash: mutable.HashMap[String, Int],
                                    val syn0: Array[Float],
                                    val syn1: Array[Float],
                                    val syn1Neg: Array[Float],
                                    val expTable: Array[Float],
                                    val unigramTable: Array[Int]) extends Serializable {

  def this(model: Map[String, Array[Float]]) = this(model, null, null, null, null, null, null, null)

  def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  /**
   * Transforms a word to its vector representation
   * @param word a word
   * @return vector representation of word
   */
  def transform(word: String): Vector = {
    model.get(word) match {
      case Some(vec) =>
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
   * Find synonyms of a word
   * @param word a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  /**
   * Find synonyms of the vector representation of a word
   * @param vector vector representation of a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    model.mapValues(vec => cosineSimilarity(fVector, vec))
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      .tail
      .toArray
  }
  
  /**
   * Returns a map of words to their vector representations.
   */
  def getVectors: Map[String, Array[Float]] = {
    model
  }
}
