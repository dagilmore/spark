package org.apache.spark.mllib.feature

import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite


class Doc2VecSuite extends FunSuite with MLlibTestSparkContext {

  test("Toy Doc2Vec Model") {
    val sentence1 = "five four three two one "
    val sentence2 = "five four four three two "
    val sentence3 = "four three three two one "
    val sentence4 = "three two two one five "
    val sentence5 = "two one one five four "

    val localDoc = Seq(sentence1, sentence2, sentence3, sentence4, sentence5)
    val doc = sc.parallelize(localDoc)
      .map(line => line.split(" ").toSeq)
    val w2v = new Word2Vec()
      .setMinCount(0)
      .setVectorSize(5)
      .setSeed(42L)
      .setNegativeSamples(0)
      .setTableSize(100)
      .fit(doc)
    val d2v = new Doc2VecModel(w2v)
      .useDBOW(useDBOW = false)
      .setNegSamples(0)
      .setMaxAlpha(0.025f)
      .setMinAlpha(0.0001f)
      .setVectorSize(5)
      .setNumIterations(5)
      .useAverage(average = true)

    val sen1vec = d2v.transform(sentence1.split(" ").toList)
    val sen2vec = d2v.transform(sentence2.split(" ").toList)
    val sen3vec = d2v.transform(sentence3.split(" ").toList)
    val sen4vec = d2v.transform(sentence4.split(" ").toList)
    val sen5vec = d2v.transform(sentence5.split(" ").toList)
  }

  test("Doc2VecModel should transform sentences") {
    val sentence = "a b " * 100 + "a c " * 10
    val localDoc = Seq(sentence, sentence)
    val doc = sc.parallelize(localDoc)
      .map(line => line.split(" ").toSeq)
    val w2v = new Word2Vec()
      .setVectorSize(10)
      .setTableSize(5)
      .setSeed(42L)
      .setNumIterations(10)
      .setNegativeSamples(0)
      .fit(doc)
    val d2v = new Doc2VecModel(w2v)
      .useDBOW(useDBOW = false)
      .setNegSamples(0)
      .setMaxAlpha(0.025f)
      .setMinAlpha(0.0001f)
      .setVectorSize(5)
      .setNumIterations(5)
      .useAverage(average = true)
    d2v.transform(List("a","c","a", "d", "e", "f", "c","a","C"))
  }

  test("Similar sentences should be close together in vector space") {
    val sentence1 = "I love to talk about computers and to listen about computers everyday " * 10
    val sentence2 = "I love to speak about computers and to hear about computers everyday " * 10
    val sentence3 = "mudding accross country is a blast . I do it ery'day " * 10
    val sentence4 = " Country biscuits ery'day accross the plains a blast a way the mudding way" * 10

    val localDoc = Seq(sentence1, sentence2, sentence3, sentence4)
    val doc = sc.parallelize(localDoc)
      .map(line => line.split(" ").toSeq)
    val dims = 50
    val w2v = new Word2Vec()
      .setMinCount(0)
      .setVectorSize(dims)
      .setSeed(42L)
      .setNegativeSamples(0)
      .fit(doc)
    val d2v = new Doc2VecModel(w2v)
      .useDBOW(useDBOW = false)
      .setNegSamples(0)
      .setVectorSize(dims)
      .setMaxAlpha(0.025f)
      .setMinAlpha(0.0001f)
      .setNumIterations(5)
      .useAverage(average = true)

    val sen1vec = d2v.transform(sentence1.split(" ").toList)
    val sen2vec = d2v.transform(sentence2.split(" ").toList)
    val sen3vec = d2v.transform(sentence3.split(" ").toList)
    val sen4vec = d2v.transform(sentence4.split(" ").toList)

    val v1_2 = d2v.cosineSimilarity(sen1vec, sen2vec)
    val v1_3 = d2v.cosineSimilarity(sen1vec, sen3vec)
    val v1_4 = d2v.cosineSimilarity(sen1vec, sen4vec)

    assert(v1_2 > v1_3 && v1_2 > v1_4)

  }

  test("Similar sentences should be close together in vector space with negative sampling") {
    val sentence1 = "I love to talk about computers and to listen about computers everyday " * 10
    val sentence2 = "I love to speak about computers and to hear about computers everyday " * 10
    val sentence3 = "mudding accross country is a blast . I do it ery'day " * 10
    val sentence4 = " Country biscuits ery'day accross the plains a blast a way the mudding way" * 10

    val localDoc = Seq(sentence1, sentence2, sentence3, sentence4)
    val doc = sc.parallelize(localDoc)
      .map(line => line.split(" ").toSeq)
    val dims = 50
    val w2v = new Word2Vec()
      .setMinCount(0)
      .setVectorSize(dims)
      .setSeed(42L)
      .setNegativeSamples(0)
      .fit(doc)
    val d2v = new Doc2VecModel(w2v)
      .useDBOW(useDBOW = false)
      .setNegSamples(0)
      .setVectorSize(dims)
      .setMaxAlpha(0.025f)
      .setMinAlpha(0.0001f)
      .setNumIterations(5)
      .useAverage(average = true)

    val sen1vec = d2v.transform(sentence1.split(" ").toList)
    val sen2vec = d2v.transform(sentence2.split(" ").toList)
    val sen3vec = d2v.transform(sentence3.split(" ").toList)
    val sen4vec = d2v.transform(sentence4.split(" ").toList)

    val v1_2 = d2v.cosineSimilarity(sen1vec, sen2vec)
    val v1_3 = d2v.cosineSimilarity(sen1vec, sen3vec)
    val v1_4 = d2v.cosineSimilarity(sen1vec, sen4vec)

    assert(v1_2 > v1_3 && v1_2 > v1_4)
  }
}
